"""CPU golden vs NPU test for moe_grouped_matmul_swiglu_quant.

Fused int8 grouped matmul -> per-channel/per-token dequant -> SwiGLU ->
dynamic per-token int8 quant.

  Inputs
    x            int8   (M, K)      ND. M = total tokens across groups.
    weight       int8   (G, K, 2N)  ND passed (framework transforms to NZ).
    weight_scale float  (G, 2N)     per-group per-channel dequant scale.
    x_scale      float  (M,)        per-token dequant scale.
    group_list   int64  (G, 2)      each row [expertIdx, rowCount]; rows are
                 laid out consecutively (group g owns the next rowCount rows).
                 NOTE: this is NOT a cumulative-offset list; kernel reads
                 groupList[g][0]=expertIdx, groupList[g][1]=count.
  Outputs
    y            int8   (M, N)      N = (2N) / 2
    y_scale      float  (M,)        per-token quant scale.

Kernel compute chain (op_kernel split_ws.h):
    acc = x_g.int32 @ weight[e].int32                      # (cnt, 2N) int32
    deq = acc * weight_scale[e] * x_scale[row]             # fp32, per-ch*per-tok
    gate = deq[:, N:2N]; up = deq[:, :N]                   # src0=second half, src1=first half
    swi  = gate * sigmoid(gate) * up                       # SiLU gate (measured)
    value = max(|swi|, axis=-1) / 127                      # y_scale
    y = round(swi / value)  -> int8  (CAST_RINT, via fp32->half->int8)

Tolerances relaxed: matmul int32 accum + bf16/half rounding on scale &
int8 cast; scale rtol/atol=2e-2, int8 allows off-by-one.

Kernel N/K constraints (matmul baseN=256 baseK=256): use K multiple of 128,
full width 2N multiple of 256 (so N is multiple of 128).
"""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def make_group_list(counts, dtype=torch.int64):
    """Build (G, 2) group list: each row [expertIdx, rowCount].

    expertIdx == g (identity mapping); rows are consecutive.
    """
    rows = [[g, int(c)] for g, c in enumerate(counts)]
    return torch.tensor(rows, dtype=dtype)


def weight_to_nz(weight):
    """(G,K,2N) int8 ND -> FRACTAL_NZ physical layout, reshape back to 3D for the wrapper.

    aclnn expects NZ shape [E, 2N/32, K/16, 16, 32]; the wrapper forces 3D,
    so rearrange the physical data per this fractal layout then reshape back to (G,K,2N).
    """
    G, K, W = weight.shape
    K0, N0 = 16, 32
    K1, N1 = K // K0, W // N0
    nz = weight.reshape(G, K1, K0, N1, N0).permute(0, 3, 1, 2, 4).contiguous()
    return nz.reshape(G, K, W).contiguous()


def moe_grouped_matmul_swiglu_quant_golden(x, weight, weight_scale, x_scale,
                                           group_list):
    """int8 grouped matmul + dequant + swiglu + dynamic int8 quant.

    group_list is (G, 2) of [expertIdx, count]; rows are consecutive.
    """
    M, K = x.shape
    G, _, W = weight.shape          # W = 2N
    N = W // 2
    y = torch.zeros((M, N), dtype=torch.int8)
    y_scale = torch.zeros((M,), dtype=torch.float32)
    gl = group_list.tolist()
    prev = 0
    for row in gl:
        e = int(row[0])
        cnt = int(row[1])
        if cnt <= 0:
            continue
        cur = prev + cnt
        xg = x[prev:cur].to(torch.int32)            # (cnt, K)
        wg = weight[e].to(torch.int32)              # (K, 2N)
        acc = torch.matmul(xg, wg).float()          # (cnt, 2N) int matmul
        sc = weight_scale[e].float().view(1, W)     # per-channel
        pts = x_scale[prev:cur].float().view(cnt, 1)  # per-token
        deq = acc * sc * pts                       # (cnt, 2N)
        gate = deq[:, :N]                           # src0 = first half
        up = deq[:, N:]                             # src1 = second half
        # kernel SwiGLU measured as SiLU gate: dst = gate * sigmoid(gate) * up
        # (on controlled data diag19 y/scale match exactly: gate=first half, up=second half, sigmoid(+gate))
        swi = gate * torch.sigmoid(gate) * up       # SiLU gate
        value = swi.abs().amax(dim=-1) / 127.0      # (cnt,)  = y_scale
        yg = torch.round(swi / value.unsqueeze(-1)).clamp(-128, 127)
        y[prev:cur] = yg.to(torch.int8)
        y_scale[prev:cur] = value
        prev = cur
    return y, y_scale


# (counts_per_group, K, 2N). G = len(counts); M = sum(counts).
# K multiple of 128; 2N multiple of 256 -> N multiple of 128.
CASES = [
    ([16, 16], 128, 256),
    ([16], 256, 256),
    ([16, 16, 16], 128, 256),
    ([32, 32], 256, 512),
    ([16, 16], 128, 512),
    ([16, 16, 16, 16], 256, 256),
    ([48], 128, 256),
    ([16, 32], 384, 256),
    ([32], 512, 512),
    ([16, 16, 16], 256, 768),
]


def _assert_quant_close(y, golden_y, scale, golden_scale):
    torch.testing.assert_close(scale.float(), golden_scale.float(),
                               rtol=2e-2, atol=2e-2)
    diff = (y.to(torch.int32) - golden_y.to(torch.int32)).abs()
    max_diff = int(diff.max()) if diff.numel() else 0
    bad_ratio= float((diff > 1).float().mean()) if diff.numel() else 0.0
    assert max_diff <= 2, f"int8 max diff {max_diff} > 2"
    assert bad_ratio < 5e-3, f"int8 diff>1 ratio {bad_ratio} too high"


@pytest.mark.parametrize("counts, K, W", CASES)
def test_moe_grouped_matmul_swiglu_quant(counts, K, W):
    torch.manual_seed(0)
    G = len(counts)
    M = sum(counts)

    x = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    weight = torch.randint(-8, 8, (G, K, W), dtype=torch.int8)
    # small scale so fp32 magnitude is reasonable / half-friendly.
    weight_scale = (torch.randn(G, W) * 0.01).to(torch.float32)
    x_scale = (torch.rand(M) * 0.5 + 0.75).to(torch.float32)
    group_list = make_group_list(counts, dtype=torch.int64)

    golden_y, golden_scale = moe_grouped_matmul_swiglu_quant_golden(
        x, weight, weight_scale, x_scale, group_list)

    # aclnn only reinterprets weight (no data move), so it must be fed in NZ physical layout.
    # NZ int8 fractal: (G,K,2N) -> (G, 2N/32, K/16, 16, 32) -> reshape back to 3D.
    weight_nz = weight_to_nz(weight)

    y_npu, scale_npu = custom_ops.moe_grouped_matmul_swiglu_quant_npu(
        x.npu(), weight_nz.npu(), weight_scale.npu(), x_scale.npu(),
        group_list.npu())

    _assert_quant_close(y_npu.cpu(), golden_y, scale_npu.cpu(), golden_scale)


def _chunk_group_pairs(group_pairs: list[tuple[int, int]], start: int, stop: int) -> list[tuple[int, int]]:
    """Intersect consecutive group spans with one chunk, preserving expert IDs."""
    if any(expert < 0 or count < 0 for expert, count in group_pairs):
        raise ValueError("Expert IDs and group counts must be nonnegative")
    if start < 0 or stop < start or stop > sum(count for _, count in group_pairs):
        raise ValueError("Chunk must be inside the concatenated group rows")
    result = []
    offset = 0
    for expert, count in group_pairs:
        end = offset + count
        overlap = max(0, min(stop, end) - max(start, offset))
        if overlap > 0:
            result.append((expert, overlap))
        offset = end
    # Non-split kernels stop at a zero count. Omit empty intersections only
    # in the reference; the full split call still exercises zero-count groups.
    return result


def test_chunk_group_pairs_preserves_experts_and_skips_empty_spans() -> None:
    pairs = [(3, 0), (2, 4), (0, 0), (1, 5), (4, 0)]
    assert _chunk_group_pairs(pairs, 0, 3) == [(2, 3)]
    assert _chunk_group_pairs(pairs, 3, 7) == [(2, 1), (1, 3)]
    assert _chunk_group_pairs(pairs, 7, 9) == [(1, 2)]
    assert _chunk_group_pairs(pairs, 0, 9) == [(2, 4), (1, 5)]
    assert _chunk_group_pairs(pairs, 4, 4) == []


def _chunked_moe_quant_reference(
    x: torch.Tensor,
    weight_nz: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_pairs: list[tuple[int, int]],
    chunk_rows: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the same public NPU operator below its workspace-split threshold."""
    if not 0 < chunk_rows <= 4096:
        raise ValueError("Reference chunks must contain at most 4096 rows")
    rows = x.shape[0]
    reference_y = torch.empty((rows, weight_scale.shape[1] // 2), dtype=torch.int8)
    reference_scale = torch.empty((rows,), dtype=torch.float32)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        pairs = _chunk_group_pairs(group_pairs, start, stop)
        assert sum(count for _, count in pairs) == stop - start
        chunk_groups = torch.tensor(pairs, dtype=torch.int64, device=x.device)
        chunk_y, chunk_scale = custom_ops.moe_grouped_matmul_swiglu_quant_npu(
            x[start:stop], weight_nz, weight_scale, x_scale[start:stop], chunk_groups
        )
        reference_y[start:stop].copy_(chunk_y.cpu())
        reference_scale[start:stop].copy_(chunk_scale.cpu())
    return reference_y, reference_scale


@pytest.mark.parametrize(
    "rows,routing",
    [
        (65536, "identity"),
        (65537, "identity"),
        (65536, "permuted"),
        (65537, "permuted"),
        (65537, "permuted_zeros"),
    ],
)
def test_moe_grouped_matmul_swiglu_quant_workspace_split(rows: int, routing: str) -> None:
    # K=128/full W=256 gives mLimit=32768 and selects workspace splitting
    # only above 65536 rows. At least one expert spans a workspace window.
    torch.manual_seed(20261015)
    hidden_size, full_width = 128, 256
    if routing == "identity":
        pairs = [(0, 40000), (1, rows - 40000)]
    elif routing == "permuted":
        pairs = [(2, 15000), (0, 25000), (3, 10000), (1, rows - 50000)]
    else:
        assert routing == "permuted_zeros" and rows > 65536
        pairs = [(2, 0), (0, 40000), (3, 0), (1, rows - 40000)]
    experts = max(expert for expert, _ in pairs) + 1
    x = torch.randint(-8, 8, (rows, hidden_size), dtype=torch.int8).npu()
    weight = torch.randint(-8, 8, (experts, hidden_size, full_width), dtype=torch.int8)
    weight_nz = weight_to_nz(weight).npu()
    weight_scale = (torch.randn(experts, full_width) * 0.01).to(torch.float32).npu()
    x_scale = (torch.rand(rows) * 0.5 + 0.75).to(torch.float32).npu()
    groups = torch.tensor(pairs, dtype=torch.int64).npu()

    actual_y, actual_scale = custom_ops.moe_grouped_matmul_swiglu_quant_npu(x, weight_nz, weight_scale, x_scale, groups)
    reference_y, reference_scale = _chunked_moe_quant_reference(x, weight_nz, weight_scale, x_scale, pairs)
    actual_y, actual_scale = actual_y.cpu(), actual_scale.cpu()
    assert actual_y.dtype == reference_y.dtype == torch.int8
    assert actual_scale.dtype == reference_scale.dtype == torch.float32
    assert torch.isfinite(actual_scale).all()
    assert torch.isfinite(reference_scale).all()
    assert (actual_scale > 0).all()
    assert (reference_scale > 0).all()
    _assert_quant_close(actual_y, reference_y, actual_scale, reference_scale)

"""Preserve native softmax while avoiding full-matrix safety temporaries."""
import triton
import triton.language as tl


@triton.jit
def _repair_all_negative_inf(scores, probabilities, n: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    first = tl.load(probabilities + row * n)
    # All -inf inputs produce NaNs from ordinary softmax. Finite rows need
    # only this first-value check. Other NaN/+inf rows retain native NaNs.
    if first != first:
        col = tl.arange(0, block)
        val = tl.load(scores + row * n + col, col < n, other=-float("inf"))
        different = tl.sum((val != -float("inf")).to(tl.int32), 0)
        if different == 0:
            tl.store(probabilities + row * n + col, 0.0, col < n)


def repair_all_negative_inf_rows_(scores, probabilities):
    n = scores.shape[-1]
    _repair_all_negative_inf[(scores.numel() // n,)](
        scores, probabilities, n, triton.next_power_of_2(n),
        num_warps=4, num_stages=1)

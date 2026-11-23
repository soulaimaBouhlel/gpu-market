"""
Step 3 — Freivalds' check (reference implementation of the on-chain check).

Claim to verify, for a batch of k items at one layer:
        W · X + b  ==  ACC        (W: n_out x n_in, X: n_in x k, ACC: n_out x k)

Instead of recomputing W·X, pick a random vector r (n_out entries) and compare
        (rᵀW) · x_t + rᵀb   vs   rᵀ · acc_t        for every item t

All arithmetic is modulo the prime P = 2^61 - 1. If ACC is wrong, the check
passes with probability <= 1/P (about 4e-19), so one round is enough.

r_i = keccak256(seed | uint8 layer | uint32 i) mod P   (same formula on-chain)
"""
from eth_utils import keccak

P = 2**61 - 1


def field(v):
    return v % P   # Python % always returns a non-negative result


def random_vector(seed, layer, n):
    return [
        int.from_bytes(keccak(seed + layer.to_bytes(1, "big") + i.to_bytes(4, "big")), "big") % P
        for i in range(n)
    ]


def check(W, b, xs, accs, seed, layer):
    """W: list of rows, b: list, xs: list of k input vectors, accs: list of k acc vectors."""
    n_out, n_in = len(W), len(W[0])
    r = random_vector(seed, layer, n_out)

    # rᵀW  (n_in values) and rᵀb (one value): computed ONCE for the whole batch
    rW = [0] * n_in
    rb = 0
    for i in range(n_out):
        for j in range(n_in):
            rW[j] = (rW[j] + r[i] * field(W[i][j])) % P
        rb = (rb + r[i] * field(b[i])) % P

    for x, acc in zip(xs, accs):
        left = rb
        for j in range(n_in):
            left = (left + rW[j] * field(x[j])) % P
        right = 0
        for i in range(n_out):
            right = (right + r[i] * field(acc[i])) % P
        if left != right:
            return False
    return True


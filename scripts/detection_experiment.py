"""
Evaluation — how often is a PARTIAL cheater caught?

A worker fakes f of the n items. The contract checks k distinct random items
(same selection rule as JobMarket.triggerChallenge). The cheater is caught
if at least one challenged item is faked:

    P(caught) = 1 - C(n - f, k) / C(n, k)      (~ 1 - (1 - F)^k for large n)

This script compares that formula with a simulation of the contract's own
selection algorithm, and checks the economics (is cheating profitable?).

Run:
    python3 scripts/detection_experiment.py
"""
import random
from math import comb

from eth_abi.packed import encode_packed
from eth_utils import keccak

K = 5                 # JobMarket.NUM_CHALLENGES
STAKE_MULTIPLIER = 10
TRIALS = 4000


def contract_selection(seed, n, k):
    """Python copy of the item selection in JobMarket.triggerChallenge."""
    items, counter = [], 0
    while len(items) < min(k, n):
        idx = int.from_bytes(keccak(encode_packed(["bytes32", "uint256"], [seed, counter])), "big") % n
        counter += 1
        if idx not in items:
            items.append(idx)
    return items


def main():
    rng = random.Random(7)
    print(f"k = {K} challenged items per job, stake = {STAKE_MULTIPLIER} x price\n")
    print(f"{'n':>4} {'faked':>6} {'F':>5} | {'formula':>8} {'simulated':>9} | {'E[gain] / price':>15}")
    print("-" * 62)
    for n in (10, 50, 200):
        seen = set()
        for F in (0.02, 0.05, 0.1, 0.2, 0.5):
            f = max(1, round(F * n))
            if f in seen:
                continue
            seen.add(f)
            exact = 1 - comb(n - f, K) / comb(n, K)
            caught = 0
            for _ in range(TRIALS):
                faked = set(rng.sample(range(n), f))
                seed = rng.randbytes(32)
                if faked & set(contract_selection(seed, n, K)):
                    caught += 1
            sim = caught / TRIALS
            # cheater saves ~F of the compute cost (assume cost ~ price) when not caught,
            # loses the stake when caught
            gain = (1 - exact) * (f / n) - exact * STAKE_MULTIPLIER
            print(f"{n:>4} {f:>6} {f / n:>5.2f} | {exact:>8.3f} {sim:>9.3f} | {gain:>15.2f}")
        print()
    print("A negative expected gain means cheating is irrational.")


if __name__ == "__main__":
    main()

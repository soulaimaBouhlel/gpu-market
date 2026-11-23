"""
Generate JSON test vectors from the PYTHON reference implementation.
The Hardhat tests replay them against the SOLIDITY implementation:
if both sides agree on every vector, the on-chain verifier is correct.

Outputs:
  fixtures/freivalds.json   random Freivalds cases (correct + tampered)
  fixtures/merkle.json      random trees and proofs
  fixtures/job_<mode>.json  a complete job (model, inputs, all proofs) per cheat mode

Run from the project root:
    python3 scripts/gen_fixtures.py
"""
import json
import os
import random
import sys

import numpy as np
from eth_utils import keccak

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.protocol import JobCommitments, model_dims, run_job  # noqa: E402
from crypto import freivalds  # noqa: E402
from crypto.encoding import int8_bytes, int32_bytes  # noqa: E402
from crypto.merkle import MerkleTree  # noqa: E402
from ml.infer_int8 import load_model, quantize_input  # noqa: E402

OUT = "fixtures"
NUM_ITEMS = 8


def hx(b):
    return "0x" + b.hex()


def gen_freivalds(rng):
    cases = []
    for t in range(40):
        n_out, n_in, k = rng.randint(1, 12), rng.randint(1, 12), rng.randint(1, 4)
        W = [[rng.randint(-127, 127) for _ in range(n_in)] for _ in range(n_out)]
        b = [rng.randint(-(2**20), 2**20) for _ in range(n_out)]
        xs = [[rng.randint(-128, 127) for _ in range(n_in)] for _ in range(k)]
        accs = [[sum(W[i][j] * x[j] for j in range(n_in)) + b[i] for i in range(n_out)] for x in xs]
        if t % 2 == 1:                                   # tamper every other case
            accs[rng.randrange(k)][rng.randrange(n_out)] += rng.choice([-1, 1]) * rng.randint(1, 99)
        seed = keccak(f"seed-{t}".encode())
        layer = rng.randint(1, 3)
        cases.append({
            "W": hx(int8_bytes([v for row in W for v in row])), "b": hx(int32_bytes(b)),
            "nIn": n_in, "nOut": n_out,
            "xs": [hx(int8_bytes(x)) for x in xs], "accs": [hx(int32_bytes(a)) for a in accs],
            "seed": hx(seed), "layer": layer,
            "expected": freivalds.check(W, b, xs, accs, seed, layer),
        })
    return cases


def gen_merkle(rng):
    cases = []
    for n in [1, 2, 3, 5, 8, 13]:
        leaves = [keccak(f"leaf-{n}-{i}".encode()) for i in range(n)]
        t = MerkleTree(leaves)
        for i in range(n):
            cases.append({"leaf": hx(leaves[i]), "index": i,
                          "proof": [hx(p) for p in t.proof(i)], "root": hx(t.root)})
    return cases


def gen_job(model, inputs, cheat):
    traces = run_job(model, inputs, cheat=cheat, fraction=1.0 if cheat == "partial" else 0.3)
    c = JobCommitments(model, inputs, traces)
    L = c.L
    # per-layer reveal data for EVERY item; the test picks the challenged ones
    layers = []
    for layer in range(1, L + 1):
        r = c.layer_reveal(list(range(len(inputs))), layer)
        layers.append({
            "W": hx(r[0]), "b": hx(r[1]), "M": r[2], "s": r[3],
            "weightsProof": [hx(p) for p in r[4]],
            "xs": [hx(x) for x in r[5]], "xProofs": [[hx(p) for p in pr] for pr in r[6]],
            "ys": [hx(y) for y in r[7]], "yProofs": [[hx(p) for p in pr] for pr in r[8]],
            "accs": [hx(a) for a in r[9]],
        })
    return {
        "cheat": cheat,
        "dims": model_dims(model),
        "weightsRoot": hx(c.weights_tree.root),
        "inputRoot": hx(c.input_tree.root),
        "activationRoot": hx(c.activation_tree.root),
        "numItems": len(inputs),
        "layers": layers,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(42)
    with open(f"{OUT}/freivalds.json", "w") as f:
        json.dump(gen_freivalds(rng), f)
    with open(f"{OUT}/merkle.json", "w") as f:
        json.dump(gen_merkle(rng), f)

    model = load_model()
    X_test = np.load("data/float_model.npz")["X_test"]
    inputs = [quantize_input(x) for x in X_test[:NUM_ITEMS]]
    for cheat in ["none", "partial", "last-layer", "swap-input"]:
        with open(f"{OUT}/job_{cheat}.json", "w") as f:
            json.dump(gen_job(model, inputs, cheat), f)
    print(f"[fixtures] written to {OUT}/")


if __name__ == "__main__":
    main()

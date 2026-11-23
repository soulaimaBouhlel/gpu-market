"""
Step 1c — Deterministic integer-only inference engine.

This is the code the WORKER runs. It uses only integer arithmetic, so every
machine produces bit-for-bit identical results.

For each layer l:
    acc = W_l @ x + b_l                                   (exact, int64)
    y   = clamp((acc * M_l + 2^(s_l - 1)) >> s_l, lo, hi) (integer requantization)
          hidden layers: [0, 127]   (this clamp IS the ReLU)
          last layer:    [-128, 127]

The smart contract re-checks exactly these formulas, so this file and
contracts/lib/LayerVerifier.sol must stay in sync.

Run from the project root (evaluates accuracy + prints a determinism hash):
    python3 ml/infer_int8.py
"""
import hashlib
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "data", "model.json")


def load_model(path=MODEL_PATH):
    with open(path) as f:
        model = json.load(f)
    for layer in model["layers"]:
        layer["W_np"] = np.array(layer["W"], dtype=np.int64)
        layer["b_np"] = np.array(layer["b"], dtype=np.int64)
    return model


def quantize_input(x_float):
    """float pixels in [0, 1] -> int8 values in [0, 127]"""
    return [int(v) for v in np.clip(np.rint(np.asarray(x_float) * 127), 0, 127)]


def requantize(acc, M, s, lo, hi):
    """Integer-only rescale. `>>` on int64 is an arithmetic shift (floor),
    exactly like Solidity's `>>` on int256."""
    acc = np.asarray(acc, dtype=np.int64)
    y = (acc * M + (1 << (s - 1))) >> s
    return np.clip(y, lo, hi)


def forward(model, x_q):
    """Run one item. Returns a trace with, for every layer, the input x,
    the accumulator acc and the output y (all Python int lists)."""
    x = np.array(x_q, dtype=np.int64)
    trace = []
    for layer in model["layers"]:
        acc = layer["W_np"] @ x + layer["b_np"]
        y = requantize(acc, layer["M"], layer["s"], layer["lo"], layer["hi"])
        trace.append({
            "x": [int(v) for v in x],
            "acc": [int(v) for v in acc],
            "y": [int(v) for v in y],
        })
        x = y
    return trace


def predict(model, x_q):
    return int(np.argmax(forward(model, x_q)[-1]["y"]))


def main():
    model = load_model()
    data = np.load(os.path.join(ROOT, "data", "float_model.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    correct = 0
    h = hashlib.sha256()
    for x, label in zip(X_test, y_test):
        trace = forward(model, quantize_input(x))
        correct += int(np.argmax(trace[-1]["y"]) == label)
        for step in trace:
            h.update(json.dumps(step["y"]).encode())

    print(f"[int8] accuracy on test set: {correct / len(y_test):.4f}")
    print(f"[int8] determinism hash of all activations: {h.hexdigest()}")
    print("       -> run this on every VM: the hash MUST be identical")


if __name__ == "__main__":
    main()


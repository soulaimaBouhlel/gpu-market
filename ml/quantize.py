"""
Step 1b — Convert the float model into an int8 model.

For each layer:
    scale_w   = max|W| / 127                  W_q = round(W / scale_w)   (int8)
    acc_scale = scale_in * scale_w            b_q = round(b / acc_scale) (int32)
    scale_out = calibrated max activation / 127
    M / 2^SHIFT ~= acc_scale / scale_out      (integer multiplier + shift)

Input pixels in [0, 1] are mapped to [0, 127], so scale_in(layer 0) = 1/127.

Output: data/model.json   (everything the worker AND the contract need)

Run from the project root:
    python3 ml/quantize.py
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "data", "float_model.npz")
OUT = os.path.join(ROOT, "data", "model.json")
SHIFT = 24


def main():
    data = np.load(IN)
    L = int(data["num_layers"])
    W = [data[f"W{i}"] for i in range(L)]
    b = [data[f"b{i}"] for i in range(L)]
    calib = data["X_train"][:500]

    # float forward pass on the calibration set -> max activation per layer
    act = calib
    act_max = []
    for i in range(L):
        z = act @ W[i].T + b[i]
        if i < L - 1:
            z = np.maximum(z, 0)              # ReLU
        act_max.append(float(np.max(np.abs(z))))
        act = z

    scale_in = 1.0 / 127.0
    layers = []
    for i in range(L):
        last = i == L - 1
        scale_w = float(np.max(np.abs(W[i]))) / 127.0
        W_q = np.clip(np.rint(W[i] / scale_w), -127, 127).astype(np.int64)
        acc_scale = scale_in * scale_w
        b_q = np.rint(b[i] / acc_scale).astype(np.int64)
        assert np.all(np.abs(b_q) < 2**31), "bias does not fit in int32"

        scale_out = act_max[i] / 127.0
        M = int(round(acc_scale / scale_out * (1 << SHIFT)))
        assert 0 < M < 2**32, "multiplier does not fit in uint32"

        layers.append({
            "n_in": int(W_q.shape[1]),
            "n_out": int(W_q.shape[0]),
            "W": W_q.tolist(),
            "b": b_q.tolist(),
            "M": M,
            "s": SHIFT,
            "lo": -128 if last else 0,     # hidden: clamp [0,127] == ReLU
            "hi": 127,
        })
        print(f"[quantize] layer {i + 1}: {W_q.shape[1]} -> {W_q.shape[0]}, M={M}, s={SHIFT}")
        scale_in = scale_out

    with open(OUT, "w") as f:
        json.dump({"layers": layers}, f)
    print(f"[quantize] saved {OUT}")


if __name__ == "__main__":
    main()


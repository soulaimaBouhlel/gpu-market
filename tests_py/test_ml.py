"""Step 1 tests — the int8 engine must be accurate AND deterministic."""
import hashlib
import json
import numpy as np

from ml.infer_int8 import load_model, forward, quantize_input, requantize


def test_requantize_matches_floor_shift():
    # arithmetic shift must round toward -infinity, like Solidity int256 >>
    assert requantize([-5], M=1, s=1, lo=-128, hi=127)[0] == (-5 + 1) >> 1 == -2
    assert requantize([5], M=1, s=1, lo=-128, hi=127)[0] == 3


def test_hidden_layers_are_relu_clamped():
    model = load_model()
    data = np.load("data/float_model.npz")
    trace = forward(model, quantize_input(data["X_test"][0]))
    for step in trace[:-1]:
        assert min(step["y"]) >= 0 and max(step["y"]) <= 127


def test_accuracy_is_close_to_float():
    model = load_model()
    data = np.load("data/float_model.npz")
    X, y = data["X_test"], data["y_test"]
    preds = [int(np.argmax(forward(model, quantize_input(x))[-1]["y"])) for x in X]
    assert np.mean(np.array(preds) == y) > 0.93


def test_forward_is_deterministic():
    model = load_model()
    data = np.load("data/float_model.npz")
    def run():
        h = hashlib.sha256()
        for x in data["X_test"][:50]:
            for step in forward(model, quantize_input(x)):
                h.update(json.dumps(step).encode())
        return h.hexdigest()
    assert run() == run()


def test_acc_is_exact_integer_matmul():
    model = load_model()
    data = np.load("data/float_model.npz")
    trace = forward(model, quantize_input(data["X_test"][3]))
    layer = model["layers"][0]
    x = trace[0]["x"]
    for i in range(layer["n_out"]):
        expected = sum(layer["W"][i][j] * x[j] for j in range(layer["n_in"])) + layer["b"][i]
        assert trace[0]["acc"][i] == expected


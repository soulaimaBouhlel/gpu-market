"""
Shared protocol logic: running a job (honestly or with a cheat mode) and
building the `LayerReveal` payload expected by JobMarket.revealLayer().

Used by agents/worker.py and scripts/gen_fixtures.py.
"""
import random

from crypto.commitments import (build_activation_tree, build_input_tree,
                                build_weights_tree)
from crypto.encoding import activation_index, int8_bytes, int32_bytes, weights_blob
from ml.infer_int8 import forward

CHEAT_MODES = ["none", "random", "partial", "last-layer", "swap-input", "silent", "no-commit"]


def _fake_outputs(trace, model, rng, from_layer=1):
    """Replace outputs from `from_layer` on with random values (acc left as computed)."""
    fake = [dict(step) for step in trace]
    for l in range(from_layer, len(model["layers"]) + 1):
        layer = model["layers"][l - 1]
        fake[l - 1]["y"] = [rng.randint(layer["lo"], layer["hi"]) for _ in range(layer["n_out"])]
        if l < len(model["layers"]):
            fake[l]["x"] = fake[l - 1]["y"]
    return fake


def run_job(model, inputs, cheat="none", fraction=0.3, seed=0):
    """Compute the traces the worker will commit to.

    cheat modes:
      none        honest computation
      random      every item's outputs are random (no real work)
      partial     a `fraction` of the items get random outputs
      last-layer  hidden layers honest, final outputs random
      swap-input  honest computation, but on the WRONG inputs
    """
    rng = random.Random(seed)
    if cheat == "swap-input":
        inputs = [list(reversed(x)) for x in inputs]

    traces = [forward(model, x) for x in inputs]
    n = len(traces)

    if cheat == "random":
        traces = [_fake_outputs(t, model, rng) for t in traces]
    elif cheat == "partial":
        faked = set(rng.sample(range(n), max(1, round(fraction * n))))
        traces = [_fake_outputs(t, model, rng) if i in faked else t for i, t in enumerate(traces)]
    elif cheat == "last-layer":
        L = len(model["layers"])
        traces = [_fake_outputs(t, model, rng, from_layer=L) for t in traces]
    return traces


class JobCommitments:
    """All three trees for one job, ready to produce proofs."""

    def __init__(self, model, inputs, traces):
        self.model = model
        self.inputs = inputs
        self.traces = traces
        self.L = len(model["layers"])
        self.weights_tree = build_weights_tree(model)
        self.input_tree = build_input_tree(inputs)
        self.activation_tree = build_activation_tree(traces)

    def layer_reveal(self, items, layer):
        """Tuple in the exact field order of the Solidity struct LayerReveal."""
        L = self.L
        ld = self.model["layers"][layer - 1]
        W_bytes, b_bytes = weights_blob(ld)
        xs, x_proofs, ys, y_proofs, accs = [], [], [], [], []
        for item in items:
            step = self.traces[item][layer - 1]
            if layer == 1:
                x = self.inputs[item]
                x_proof = self.input_tree.proof(item)
            else:
                x = self.traces[item][layer - 2]["y"]
                x_proof = self.activation_tree.proof(activation_index(item, layer - 1, L))
            xs.append(int8_bytes(x))
            x_proofs.append(x_proof)
            ys.append(int8_bytes(step["y"]))
            y_proofs.append(self.activation_tree.proof(activation_index(item, layer, L)))
            accs.append(int32_bytes(step["acc"]))
        return (
            W_bytes, b_bytes, int(ld["M"]), int(ld["s"]),
            self.weights_tree.proof(layer - 1),
            xs, x_proofs, ys, y_proofs, accs,
        )


def model_dims(model):
    return [model["layers"][0]["n_in"]] + [l["n_out"] for l in model["layers"]]

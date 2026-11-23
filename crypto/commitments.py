"""
Step 2c — Build the three Merkle trees used by the protocol.

    input tree       (built by the RENTER)  leaves: one per item
    activation tree  (built by the WORKER)  leaves: one per (item, layer)
    weights tree     (built once per MODEL) leaves: one per layer
"""
from crypto.encoding import input_leaf, activation_leaf, weights_leaf
from crypto.merkle import MerkleTree


def build_input_tree(inputs):
    return MerkleTree([input_leaf(i, x) for i, x in enumerate(inputs)])


def build_activation_tree(traces):
    """traces[item][layer-1]['y'] -> leaf at index item * L + (layer - 1)"""
    leaves = []
    for item, trace in enumerate(traces):
        for l, step in enumerate(trace, start=1):
            leaves.append(activation_leaf(item, l, step["y"]))
    return MerkleTree(leaves)


def build_weights_tree(model):
    return MerkleTree([weights_leaf(l, layer) for l, layer in enumerate(model["layers"], start=1)])


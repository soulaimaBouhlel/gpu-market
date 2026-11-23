"""
Step 2a — Byte encodings and leaf hashes.

These formats are a CONTRACT between Python and Solidity: both sides must
hash exactly the same bytes. Never change one side without the other.

    int8 tensor   -> 1 byte per value, two's complement
    int32 vector  -> 4 bytes per value, big-endian two's complement

    input leaf      = keccak256(0x00 | uint32 item | int8 bytes(x))
    activation leaf = keccak256(0x00 | uint32 item | uint8 layer | int8 bytes(y))
    weights leaf    = keccak256(0x00 | uint8 layer | int8 bytes(W row-major)
                                     | int32 bytes(b) | uint32 M | uint8 s)

The 0x00 prefix on leaves (and 0x01 on internal nodes, see merkle.py) is
"domain separation": it makes it impossible to pass an internal node off
as a leaf (second-preimage attack).
"""
from eth_utils import keccak


def int8_bytes(values):
    out = bytearray()
    for v in values:
        v = int(v)
        if not -128 <= v <= 127:
            raise ValueError(f"value {v} does not fit in int8")
        out += (v & 0xFF).to_bytes(1, "big")
    return bytes(out)


def int32_bytes(values):
    out = bytearray()
    for v in values:
        v = int(v)
        if not -(2**31) <= v < 2**31:
            raise ValueError(f"value {v} does not fit in int32")
        out += (v & 0xFFFFFFFF).to_bytes(4, "big")
    return bytes(out)


def decode_int8(data):
    return [b - 256 if b > 127 else b for b in data]


def decode_int32(data):
    vals = []
    for i in range(0, len(data), 4):
        v = int.from_bytes(data[i:i + 4], "big")
        vals.append(v - 2**32 if v >= 2**31 else v)
    return vals


def input_leaf(item, x):
    return keccak(b"\x00" + item.to_bytes(4, "big") + int8_bytes(x))


def activation_leaf(item, layer, y):
    return keccak(b"\x00" + item.to_bytes(4, "big") + layer.to_bytes(1, "big") + int8_bytes(y))


def weights_blob(layer_dict):
    """Flattened W (row-major) and b, as bytes."""
    W_flat = [v for row in layer_dict["W"] for v in row]
    return int8_bytes(W_flat), int32_bytes(layer_dict["b"])


def weights_leaf(layer, layer_dict):
    W_bytes, b_bytes = weights_blob(layer_dict)
    return keccak(
        b"\x00"
        + layer.to_bytes(1, "big")
        + W_bytes
        + b_bytes
        + int(layer_dict["M"]).to_bytes(4, "big")
        + int(layer_dict["s"]).to_bytes(1, "big")
    )


def activation_index(item, layer, num_layers):
    """Position of (item, layer) in the worker's activation tree. layer is 1-based."""
    return item * num_layers + (layer - 1)

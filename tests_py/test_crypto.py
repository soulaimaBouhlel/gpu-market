"""Steps 2 & 3 tests — Merkle trees and Freivalds."""
import random

from eth_utils import keccak

from crypto import freivalds
from crypto.encoding import (activation_leaf, decode_int8, decode_int32,
                             input_leaf, int8_bytes, int32_bytes)
from crypto.merkle import MerkleTree, verify


# ---------- encoding ----------
def test_int8_roundtrip():
    vals = [-128, -1, 0, 1, 127]
    assert decode_int8(int8_bytes(vals)) == vals


def test_int32_roundtrip():
    vals = [-(2**31), -1, 0, 1, 2**31 - 1]
    assert decode_int32(int32_bytes(vals)) == vals


# ---------- merkle ----------
def test_all_proofs_verify_for_many_sizes():
    for n in range(1, 40):
        leaves = [keccak(str(i).encode()) for i in range(n)]
        t = MerkleTree(leaves)
        for i, leaf in enumerate(leaves):
            assert verify(leaf, i, t.proof(i), t.root)


def test_tampered_leaf_fails():
    leaves = [input_leaf(i, [i, 2, 3]) for i in range(8)]
    t = MerkleTree(leaves)
    fake = input_leaf(5, [5, 2, 4])                 # one value changed
    assert not verify(fake, 5, t.proof(5), t.root)


def test_right_data_wrong_position_fails():
    leaves = [activation_leaf(i, 1, [1, 2, 3]) for i in range(8)]
    t = MerkleTree(leaves)
    # the leaf hash includes the item number, so item 3's data can't be used as item 2
    assert not verify(activation_leaf(3, 1, [1, 2, 3]), 2, t.proof(2), t.root)


# ---------- freivalds ----------
def _random_layer(rng, n_out, n_in, k):
    W = [[rng.randint(-127, 127) for _ in range(n_in)] for _ in range(n_out)]
    b = [rng.randint(-50000, 50000) for _ in range(n_out)]
    xs = [[rng.randint(0, 127) for _ in range(n_in)] for _ in range(k)]
    accs = [[sum(W[i][j] * x[j] for j in range(n_in)) + b[i] for i in range(n_out)] for x in xs]
    return W, b, xs, accs


def test_freivalds_accepts_correct():
    rng = random.Random(1)
    for t in range(20):
        W, b, xs, accs = _random_layer(rng, 16, 32, 5)
        assert freivalds.check(W, b, xs, accs, keccak(bytes([t])), 1)


def test_freivalds_catches_single_wrong_entry():
    rng = random.Random(2)
    for t in range(200):
        W, b, xs, accs = _random_layer(rng, 8, 12, 3)
        accs[rng.randrange(3)][rng.randrange(8)] += rng.choice([-1, 1]) * rng.randint(1, 1000)
        assert not freivalds.check(W, b, xs, accs, keccak(bytes([t % 256, t // 256])), 1)

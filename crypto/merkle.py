"""
Step 2b — Merkle tree (keccak256, compatible with contracts/lib/LayerVerifier.sol).

    internal node = keccak256(0x01 | left | right)
    odd number of nodes on a level -> the last node is paired with itself

    proof = list of sibling hashes from the leaf up to the root.
    At each level, bit 0 of the index says where we are:
        even index -> we are the LEFT child  -> h = node(h, sibling)
        odd index  -> we are the RIGHT child -> h = node(sibling, h)
"""
from eth_utils import keccak


def node(left, right):
    return keccak(b"\x01" + left + right)


class MerkleTree:
    def __init__(self, leaves):
        if not leaves:
            raise ValueError("empty tree")
        self.levels = [list(leaves)]
        while len(self.levels[-1]) > 1:
            cur = self.levels[-1]
            nxt = []
            for i in range(0, len(cur), 2):
                left = cur[i]
                right = cur[i + 1] if i + 1 < len(cur) else cur[i]
                nxt.append(node(left, right))
            self.levels.append(nxt)

    @property
    def root(self):
        return self.levels[-1][0]

    def proof(self, index):
        proof = []
        for level in self.levels[:-1]:
            sib = index ^ 1
            proof.append(level[sib] if sib < len(level) else level[index])
            index //= 2
        return proof


def verify(leaf, index, proof, root):
    h = leaf
    for sib in proof:
        h = node(h, sib) if index % 2 == 0 else node(sib, h)
        index //= 2
    return h == root

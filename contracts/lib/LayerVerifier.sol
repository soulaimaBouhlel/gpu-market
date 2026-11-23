// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title LayerVerifier
/// @notice On-chain mirror of crypto/encoding.py, crypto/merkle.py,
///         crypto/freivalds.py and the requantization in ml/infer_int8.py.
///         Every formula here MUST match the Python reference byte for byte.
library LayerVerifier {
    /// Freivalds prime: 2^61 - 1
    uint256 internal constant P = 2305843009213693951;

    // ------------------------------------------------------------------
    // Merkle
    // ------------------------------------------------------------------

    function nodeHash(bytes32 left, bytes32 right) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(bytes1(0x01), left, right));
    }

    function verifyProof(bytes32 leaf, uint256 index, bytes32[] calldata proof, bytes32 root)
        internal pure returns (bool)
    {
        bytes32 h = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            h = (index % 2 == 0) ? nodeHash(h, proof[i]) : nodeHash(proof[i], h);
            index /= 2;
        }
        return h == root;
    }

    function inputLeaf(uint32 item, bytes calldata x) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(bytes1(0x00), item, x));
    }

    function activationLeaf(uint32 item, uint8 layer, bytes calldata y) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(bytes1(0x00), item, layer, y));
    }

    function weightsLeaf(uint8 layer, bytes calldata W, bytes calldata b, uint32 M, uint8 s)
        internal pure returns (bytes32)
    {
        return keccak256(abi.encodePacked(bytes1(0x00), layer, W, b, M, s));
    }

    // ------------------------------------------------------------------
    // Decoding packed tensors
    // ------------------------------------------------------------------

    function readInt8(bytes calldata d, uint256 i) internal pure returns (int256) {
        return int256(int8(uint8(d[i])));
    }

    function readInt32(bytes calldata d, uint256 i) internal pure returns (int256) {
        return int256(int32(uint32(bytes4(d[i * 4:i * 4 + 4]))));
    }

    /// signed integer -> element of the field [0, P)
    function toField(int256 v) internal pure returns (uint256) {
        if (v >= 0) return uint256(v) % P;
        uint256 m = uint256(-v) % P;
        return m == 0 ? 0 : P - m;
    }

    // ------------------------------------------------------------------
    // Freivalds: check  W · x_t + b == acc_t  for every item t of the batch
    // ------------------------------------------------------------------

    function freivalds(
        bytes calldata W,          // n_out * n_in int8, row-major
        bytes calldata b,          // n_out int32
        uint256 nIn,
        uint256 nOut,
        bytes[] calldata xs,       // k inputs, n_in int8 each
        bytes[] calldata accs,     // k accumulators, n_out int32 each
        bytes32 seed,
        uint8 layer
    ) internal pure returns (bool) {
        uint256[] memory r = new uint256[](nOut);
        for (uint256 i = 0; i < nOut; i++) {
            r[i] = uint256(keccak256(abi.encodePacked(seed, layer, uint32(i)))) % P;
        }

        // rᵀW and rᵀb, computed once for the whole batch
        uint256[] memory rW = new uint256[](nIn);
        uint256 rb = 0;
        for (uint256 i = 0; i < nOut; i++) {
            uint256 ri = r[i];
            for (uint256 j = 0; j < nIn; j++) {
                rW[j] = addmod(rW[j], mulmod(ri, toField(readInt8(W, i * nIn + j)), P), P);
            }
            rb = addmod(rb, mulmod(ri, toField(readInt32(b, i)), P), P);
        }

        for (uint256 t = 0; t < xs.length; t++) {
            uint256 left = rb;
            for (uint256 j = 0; j < nIn; j++) {
                left = addmod(left, mulmod(rW[j], toField(readInt8(xs[t], j)), P), P);
            }
            uint256 right = 0;
            for (uint256 i = 0; i < nOut; i++) {
                right = addmod(right, mulmod(r[i], toField(readInt32(accs[t], i)), P), P);
            }
            if (left != right) return false;
        }
        return true;
    }

    // ------------------------------------------------------------------
    // Requantization: y == clamp((acc * M + 2^(s-1)) >> s, lo, hi)
    // ------------------------------------------------------------------

    function requantOk(
        bytes calldata acc,
        bytes calldata y,
        uint256 nOut,
        uint32 M,
        uint8 s,
        int256 lo,
        int256 hi
    ) internal pure returns (bool) {
        int256 half = int256(1) << (s - 1);
        for (uint256 i = 0; i < nOut; i++) {
            int256 v = (readInt32(acc, i) * int256(uint256(M)) + half) >> s;  // floor shift
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            if (v != readInt8(y, i)) return false;
        }
        return true;
    }
}

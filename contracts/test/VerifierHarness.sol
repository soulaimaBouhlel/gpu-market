// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../lib/LayerVerifier.sol";

/// @notice Test-only contract exposing the library functions so the
///         Hardhat tests can compare them with the Python reference.
contract VerifierHarness {
    function verifyProof(bytes32 leaf, uint256 index, bytes32[] calldata proof, bytes32 root)
        external pure returns (bool)
    {
        return LayerVerifier.verifyProof(leaf, index, proof, root);
    }

    function freivalds(
        bytes calldata W, bytes calldata b, uint256 nIn, uint256 nOut,
        bytes[] calldata xs, bytes[] calldata accs, bytes32 seed, uint8 layer
    ) external pure returns (bool) {
        return LayerVerifier.freivalds(W, b, nIn, nOut, xs, accs, seed, layer);
    }

    function requantOk(
        bytes calldata acc, bytes calldata y, uint256 nOut, uint32 M, uint8 s, int256 lo, int256 hi
    ) external pure returns (bool) {
        return LayerVerifier.requantOk(acc, y, nOut, M, s, lo, hi);
    }

    function inputLeaf(uint32 item, bytes calldata x) external pure returns (bytes32) {
        return LayerVerifier.inputLeaf(item, x);
    }

    function activationLeaf(uint32 item, uint8 layer, bytes calldata y) external pure returns (bytes32) {
        return LayerVerifier.activationLeaf(item, layer, y);
    }
}

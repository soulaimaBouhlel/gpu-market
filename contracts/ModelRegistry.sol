// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ModelRegistry
/// @notice Stores only a fingerprint of each AI model: the Merkle root of its
///         weights (one leaf per layer) and the layer dimensions.
///         The weights themselves stay off-chain and are sent in calldata
///         only when a layer has to be verified.
contract ModelRegistry {
    struct Model {
        bytes32 weightsRoot;
        uint32[] dims;     // [n_in(layer1), n_out(layer1), n_out(layer2), ...]
        address owner;
    }

    Model[] private models;

    event ModelRegistered(uint256 indexed modelId, bytes32 weightsRoot, uint256 numLayers, address owner);

    function registerModel(bytes32 weightsRoot, uint32[] calldata dims) external returns (uint256 modelId) {
        require(weightsRoot != bytes32(0), "empty root");
        require(dims.length >= 2 && dims.length <= 33, "bad dims");
        for (uint256 i = 0; i < dims.length; i++) require(dims[i] > 0, "zero dim");

        models.push(Model({weightsRoot: weightsRoot, dims: dims, owner: msg.sender}));
        modelId = models.length - 1;
        emit ModelRegistered(modelId, weightsRoot, dims.length - 1, msg.sender);
    }

    function exists(uint256 modelId) external view returns (bool) {
        return modelId < models.length;
    }

    function getModel(uint256 modelId)
        external view returns (bytes32 weightsRoot, uint32[] memory dims, uint256 numLayers)
    {
        require(modelId < models.length, "unknown model");
        Model storage m = models[modelId];
        return (m.weightsRoot, m.dims, m.dims.length - 1);
    }

    function modelCount() external view returns (uint256) {
        return models.length;
    }
}

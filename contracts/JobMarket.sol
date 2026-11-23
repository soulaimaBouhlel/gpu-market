// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./ModelRegistry.sol";
import "./lib/LayerVerifier.sol";

/// @title JobMarket
/// @notice Escrow + staking + on-chain verification of int8 neural-network inference.
///
///  OPEN ──acceptJob──► ASSIGNED ──commitResult──► COMMITTED ──triggerChallenge──► CHALLENGED
///    │                    │ (no commit in time)                                      │
///    ▼                    ▼                                       revealLayer × L    │
///  CANCELLED           SLASHED ◄──────── check fails / no reveal in time ────────────┤
///                                                                                    ▼
///                                                                                  PAID
contract JobMarket {
    using LayerVerifier for *;

    // ---------------------------------------------------------------- parameters
    uint256 public constant CHALLENGE_DELAY = 3;        // blocks between commit and seed block
    uint256 public constant NUM_CHALLENGES = 5;         // items checked per job (k)
    uint256 public constant STAKE_MULTIPLIER = 10;      // stake locked = 10 × price

    enum State { None, Open, Assigned, Committed, Challenged, Paid, Slashed, Cancelled }

    struct Job {
        address renter;
        address worker;
        uint256 modelId;
        bytes32 inputRoot;        // committed by the renter
        bytes32 activationRoot;   // committed by the worker
        uint32 numItems;
        uint256 price;
        uint256 lockedStake;
        uint64 computeWindow;
        uint64 deadline;          // meaning depends on the state
        uint256 seedBlock;
        bytes32 seed;
        uint32 layersVerified;    // bitmask: bit (l-1) set when layer l passed
        State state;
        string inputsURI;
        string resultsURI;
    }

    /// Everything the worker sends to prove one layer for the k challenged items.
    struct LayerReveal {
        bytes W;                  // int8, n_out × n_in, row-major
        bytes b;                  // int32 × n_out
        uint32 M;
        uint8 s;
        bytes32[] weightsProof;
        bytes[] xs;               // k inputs of this layer
        bytes32[][] xProofs;
        bytes[] ys;               // k outputs of this layer
        bytes32[][] yProofs;
        bytes[] accs;             // k int32 accumulators (not committed, pinned by Freivalds)
    }

    ModelRegistry public immutable registry;
    uint256 public immutable revealWindow;              // seconds to answer the challenge
    Job[] private jobs;
    mapping(uint256 => uint32[]) private challenged;

    mapping(address => uint256) public stakes;        // total stake deposited
    mapping(address => uint256) public lockedStakes;  // part locked in running jobs
    mapping(address => uint256) public balances;      // withdrawable payouts / refunds
    uint256 public treasury;

    // ---------------------------------------------------------------- events
    event StakeDeposited(address indexed node, uint256 amount);
    event StakeWithdrawn(address indexed node, uint256 amount);
    event JobCreated(uint256 indexed jobId, address indexed renter, uint256 modelId, uint32 numItems, uint256 price, string inputsURI);
    event JobCancelled(uint256 indexed jobId);
    event JobAccepted(uint256 indexed jobId, address indexed worker, uint256 lockedStake);
    event ResultCommitted(uint256 indexed jobId, bytes32 activationRoot, uint256 seedBlock, string resultsURI);
    event ChallengeWindowReset(uint256 indexed jobId, uint256 newSeedBlock);
    event ChallengeIssued(uint256 indexed jobId, uint32[] items, bytes32 seed);
    event LayerVerified(uint256 indexed jobId, uint8 layer);
    event JobPaid(uint256 indexed jobId, address indexed worker, uint256 amount);
    event JobSlashed(uint256 indexed jobId, address indexed worker, string reason);

    constructor(ModelRegistry _registry, uint256 _revealWindow) {
        require(_revealWindow > 0, "no reveal window");
        registry = _registry;
        revealWindow = _revealWindow;
    }

    // ================================================================ staking
    function depositStake() external payable {
        require(msg.value > 0, "no value");
        stakes[msg.sender] += msg.value;
        emit StakeDeposited(msg.sender, msg.value);
    }

    function freeStake(address node) public view returns (uint256) {
        return stakes[node] - lockedStakes[node];
    }

    function withdrawStake(uint256 amount) external {
        require(amount <= freeStake(msg.sender), "stake locked");
        stakes[msg.sender] -= amount;
        payable(msg.sender).transfer(amount);
        emit StakeWithdrawn(msg.sender, amount);
    }

    function withdraw() external {
        uint256 amount = balances[msg.sender];
        require(amount > 0, "nothing to withdraw");
        balances[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }

    // ================================================================ renter side
    function createJob(
        uint256 modelId,
        bytes32 inputRoot,
        uint32 numItems,
        string calldata inputsURI,
        uint64 computeWindow
    ) external payable returns (uint256 jobId) {
        require(msg.value > 0, "payment required");
        require(registry.exists(modelId), "unknown model");
        require(numItems > 0, "no items");
        require(inputRoot != bytes32(0), "empty input root");
        require(computeWindow > 0, "no compute window");

        jobs.push();
        jobId = jobs.length - 1;
        Job storage j = jobs[jobId];
        j.renter = msg.sender;
        j.modelId = modelId;
        j.inputRoot = inputRoot;
        j.numItems = numItems;
        j.price = msg.value;
        j.computeWindow = computeWindow;
        j.inputsURI = inputsURI;
        j.state = State.Open;
        emit JobCreated(jobId, msg.sender, modelId, numItems, msg.value, inputsURI);
    }

    function cancelJob(uint256 jobId) external {
        Job storage j = jobs[jobId];
        require(j.state == State.Open, "not open");
        require(msg.sender == j.renter, "not renter");
        j.state = State.Cancelled;
        balances[j.renter] += j.price;
        emit JobCancelled(jobId);
    }

    // ================================================================ worker side
    function acceptJob(uint256 jobId) external {
        Job storage j = jobs[jobId];
        require(j.state == State.Open, "not open");
        require(msg.sender != j.renter, "renter cannot work");
        uint256 lock = j.price * STAKE_MULTIPLIER;
        require(freeStake(msg.sender) >= lock, "insufficient stake");

        lockedStakes[msg.sender] += lock;
        j.worker = msg.sender;
        j.lockedStake = lock;
        j.deadline = uint64(block.timestamp) + j.computeWindow;
        j.state = State.Assigned;
        emit JobAccepted(jobId, msg.sender, lock);
    }

    function commitResult(uint256 jobId, bytes32 activationRoot, string calldata resultsURI) external {
        Job storage j = jobs[jobId];
        require(j.state == State.Assigned, "not assigned");
        require(msg.sender == j.worker, "not worker");
        require(block.timestamp <= j.deadline, "too late");
        require(activationRoot != bytes32(0), "empty root");

        j.activationRoot = activationRoot;
        j.resultsURI = resultsURI;
        j.seedBlock = block.number + CHALLENGE_DELAY;   // randomness comes AFTER the commit
        j.state = State.Committed;
        emit ResultCommitted(jobId, activationRoot, j.seedBlock, resultsURI);
    }

    /// Anyone can call this once the seed block is mined.
    function triggerChallenge(uint256 jobId) external {
        Job storage j = jobs[jobId];
        require(j.state == State.Committed, "not committed");
        require(block.number > j.seedBlock, "seed block not mined yet");

        // blockhash() only works for the 256 most recent blocks
        if (block.number - j.seedBlock > 256) {
            j.seedBlock = block.number + CHALLENGE_DELAY;
            emit ChallengeWindowReset(jobId, j.seedBlock);
            return;
        }

        bytes32 seed = keccak256(abi.encodePacked(blockhash(j.seedBlock), jobId));
        j.seed = seed;

        uint256 k = j.numItems < NUM_CHALLENGES ? j.numItems : NUM_CHALLENGES;
        uint32[] storage items = challenged[jobId];
        uint256 counter = 0;
        while (items.length < k) {
            uint32 idx = uint32(uint256(keccak256(abi.encodePacked(seed, counter))) % j.numItems);
            counter++;
            bool dup = false;
            for (uint256 t = 0; t < items.length; t++) {
                if (items[t] == idx) { dup = true; break; }
            }
            if (!dup) items.push(idx);
        }

        j.deadline = uint64(block.timestamp + revealWindow);
        j.state = State.Challenged;
        emit ChallengeIssued(jobId, items, seed);
    }

    /// Prove one layer for all challenged items. Call once per layer (any order).
    function revealLayer(uint256 jobId, uint8 layer, LayerReveal calldata r) external {
        Job storage j = jobs[jobId];
        require(j.state == State.Challenged, "not challenged");
        require(msg.sender == j.worker, "not worker");
        require(block.timestamp <= j.deadline, "reveal too late");

        (bytes32 weightsRoot, uint32[] memory dims, uint256 L) = registry.getModel(j.modelId);
        require(layer >= 1 && layer <= L, "bad layer");
        require(j.layersVerified & (uint32(1) << (layer - 1)) == 0, "layer already verified");

        uint256 nIn = dims[layer - 1];
        uint256 nOut = dims[layer];

        // ---- 1. the weights really are the registered model's weights
        require(r.W.length == nIn * nOut && r.b.length == 4 * nOut, "bad weight sizes");
        require(
            LayerVerifier.verifyProof(
                LayerVerifier.weightsLeaf(layer, r.W, r.b, r.M, r.s), layer - 1, r.weightsProof, weightsRoot
            ),
            "bad weights proof"
        );

        // ---- 2. every x and y really belongs to the committed trees
        _checkTensors(j, jobId, layer, L, nIn, nOut, r);

        // ---- 3. the computation itself
        bool ok = LayerVerifier.freivalds(r.W, r.b, nIn, nOut, r.xs, r.accs, j.seed, layer);
        if (ok) {
            int256 lo = layer == L ? int256(-128) : int256(0);   // hidden layers: ReLU clamp
            for (uint256 t = 0; t < r.accs.length && ok; t++) {
                ok = LayerVerifier.requantOk(r.accs[t], r.ys[t], nOut, r.M, r.s, lo, 127);
            }
        }
        if (!ok) {
            _slash(jobId, "computation check failed");
            return;
        }

        j.layersVerified |= uint32(1) << (layer - 1);
        emit LayerVerified(jobId, layer);

        if (j.layersVerified == uint32((uint256(1) << L) - 1)) {
            _pay(jobId);
        }
    }

    // ================================================================ timeouts
    function claimTimeout(uint256 jobId) external {
        Job storage j = jobs[jobId];
        require(block.timestamp > j.deadline, "deadline not passed");
        if (j.state == State.Assigned) {
            _slash(jobId, "no commit before deadline");
        } else if (j.state == State.Challenged) {
            _slash(jobId, "no reveal before deadline");
        } else {
            revert("nothing to claim");
        }
    }

    // ================================================================ views
    function getJob(uint256 jobId) external view returns (Job memory) {
        return jobs[jobId];
    }

    function getChallengedItems(uint256 jobId) external view returns (uint32[] memory) {
        return challenged[jobId];
    }

    function jobCount() external view returns (uint256) {
        return jobs.length;
    }

    // ================================================================ internals
    function _checkTensors(
        Job storage j, uint256 jobId, uint8 layer, uint256 L, uint256 nIn, uint256 nOut, LayerReveal calldata r
    ) private view {
        uint32[] storage items = challenged[jobId];
        uint256 k = items.length;
        require(
            r.xs.length == k && r.ys.length == k && r.accs.length == k &&
            r.xProofs.length == k && r.yProofs.length == k,
            "wrong batch size"
        );
        for (uint256 t = 0; t < k; t++) {
            uint32 item = items[t];
            require(r.xs[t].length == nIn && r.ys[t].length == nOut && r.accs[t].length == 4 * nOut, "bad tensor size");

            bool xOk = layer == 1
                // layer 1 input = the renter's data, proven against the RENTER's root
                ? LayerVerifier.verifyProof(LayerVerifier.inputLeaf(item, r.xs[t]), item, r.xProofs[t], j.inputRoot)
                // deeper layers: input = previous layer's output in the worker's tree
                : LayerVerifier.verifyProof(
                    LayerVerifier.activationLeaf(item, layer - 1, r.xs[t]),
                    uint256(item) * L + (layer - 2), r.xProofs[t], j.activationRoot);
            require(xOk, "bad input proof");

            require(
                LayerVerifier.verifyProof(
                    LayerVerifier.activationLeaf(item, layer, r.ys[t]),
                    uint256(item) * L + (layer - 1), r.yProofs[t], j.activationRoot),
                "bad output proof"
            );
        }
    }

    function _pay(uint256 jobId) private {
        Job storage j = jobs[jobId];
        j.state = State.Paid;
        lockedStakes[j.worker] -= j.lockedStake;
        balances[j.worker] += j.price;
        emit JobPaid(jobId, j.worker, j.price);
    }

    function _slash(uint256 jobId, string memory reason) private {
        Job storage j = jobs[jobId];
        j.state = State.Slashed;
        uint256 s = j.lockedStake;
        lockedStakes[j.worker] -= s;
        stakes[j.worker] -= s;
        balances[j.renter] += j.price + s / 2;   // refund + compensation
        treasury += s - s / 2;
        emit JobSlashed(jobId, j.worker, reason);
    }
}

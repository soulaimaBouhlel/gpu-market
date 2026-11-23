// JobMarket protocol tests: money flows, state machine, and on-chain
// detection of each cheat mode (fixtures from scripts/gen_fixtures.py).
const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time, mine } = require("@nomicfoundation/hardhat-network-helpers");
const fs = require("fs");

const load = (name) => JSON.parse(fs.readFileSync(`fixtures/${name}`, "utf8"));
const PRICE = ethers.parseEther("1");
const STAKE = ethers.parseEther("20");
const State = { Open: 1, Assigned: 2, Committed: 3, Challenged: 4, Paid: 5, Slashed: 6, Cancelled: 7 };

async function deploy(job) {
  const [owner, renter, worker, other] = await ethers.getSigners();
  const registry = await (await ethers.getContractFactory("ModelRegistry")).deploy();
  const market = await (await ethers.getContractFactory("JobMarket")).deploy(await registry.getAddress(), 600);
  await registry.registerModel(job.weightsRoot, job.dims);
  await market.connect(worker).depositStake({ value: STAKE });
  return { registry, market, owner, renter, worker, other };
}

async function createJob(market, renter, job) {
  await market.connect(renter).createJob(0, job.inputRoot, job.numItems, "http://storage/inputs", 3600, { value: PRICE });
  return 0;
}

async function runUntilChallenge(market, renter, worker, job) {
  const id = await createJob(market, renter, job);
  await market.connect(worker).acceptJob(id);
  await market.connect(worker).commitResult(id, job.activationRoot, "http://storage/results");
  await mine(4);
  await market.triggerChallenge(id);
  return id;
}

// Build the LayerReveal struct for the challenged items from the fixture
function reveal(job, items, layer) {
  const d = job.layers[layer - 1];
  const pick = (arr) => items.map((i) => arr[Number(i)]);
  return {
    W: d.W, b: d.b, M: d.M, s: d.s, weightsProof: d.weightsProof,
    xs: pick(d.xs), xProofs: pick(d.xProofs),
    ys: pick(d.ys), yProofs: pick(d.yProofs),
    accs: pick(d.accs),
  };
}

describe("JobMarket", function () {
  describe("staking and escrow", function () {
    it("locks the worker's stake on accept and blocks withdrawal", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      await createJob(market, renter, job);
      await market.connect(worker).acceptJob(0);
      expect(await market.lockedStakes(worker.address)).to.equal(PRICE * 10n);
      await expect(market.connect(worker).withdrawStake(STAKE)).to.be.revertedWith("stake locked");
    });

    it("refuses a worker without enough stake", async () => {
      const job = load("job_none.json");
      const { market, renter, other } = await deploy(job);
      await createJob(market, renter, job);
      await expect(market.connect(other).acceptJob(0)).to.be.revertedWith("insufficient stake");
    });

    it("lets the renter cancel an open job and get refunded", async () => {
      const job = load("job_none.json");
      const { market, renter } = await deploy(job);
      await createJob(market, renter, job);
      await market.connect(renter).cancelJob(0);
      expect(await market.balances(renter.address)).to.equal(PRICE);
      expect((await market.getJob(0)).state).to.equal(State.Cancelled);
    });
  });

  describe("state machine guards", function () {
    it("only the worker can commit, and only once assigned", async () => {
      const job = load("job_none.json");
      const { market, renter, worker, other } = await deploy(job);
      await createJob(market, renter, job);
      await expect(market.connect(worker).commitResult(0, job.activationRoot, "")).to.be.revertedWith("not assigned");
      await market.connect(worker).acceptJob(0);
      await expect(market.connect(other).commitResult(0, job.activationRoot, "")).to.be.revertedWith("not worker");
    });

    it("cannot trigger the challenge before the seed block is mined", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      await createJob(market, renter, job);
      await market.connect(worker).acceptJob(0);
      await market.connect(worker).commitResult(0, job.activationRoot, "");
      await expect(market.triggerChallenge(0)).to.be.revertedWith("seed block not mined yet");
    });

    it("picks distinct challenged items", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      const items = (await market.getChallengedItems(id)).map(Number);
      expect(items.length).to.equal(5);
      expect(new Set(items).size).to.equal(5);
    });
  });

  describe("honest worker", function () {
    it("passes all layers and gets paid", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      const items = await market.getChallengedItems(id);
      for (let l = 1; l <= job.layers.length; l++) {
        const tx = await market.connect(worker).revealLayer(id, l, reveal(job, items, l));
        const rcpt = await tx.wait();
        console.log(`      gas revealLayer(${l}): ${rcpt.gasUsed}`);
      }
      const j = await market.getJob(id);
      expect(j.state).to.equal(State.Paid);
      expect(await market.balances(worker.address)).to.equal(PRICE);
      expect(await market.lockedStakes(worker.address)).to.equal(0n);
    });

    it("rejects revealing the same layer twice", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      const items = await market.getChallengedItems(id);
      await market.connect(worker).revealLayer(id, 1, reveal(job, items, 1));
      await expect(market.connect(worker).revealLayer(id, 1, reveal(job, items, 1)))
        .to.be.revertedWith("layer already verified");
    });
  });

  describe("cheating workers", function () {
    it("fake accumulator -> Freivalds fails -> slashed", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      const items = await market.getChallengedItems(id);
      const r = reveal(job, items, 1);
      const acc = ethers.getBytes(r.accs[0]);
      acc[3] ^= 1;                                   // change one accumulator value
      r.accs = [ethers.hexlify(acc), ...r.accs.slice(1)];
      await expect(market.connect(worker).revealLayer(id, 1, r))
        .to.emit(market, "JobSlashed").withArgs(id, worker.address, "computation check failed");
      // renter gets the price back + half of the slashed stake
      expect(await market.balances(renter.address)).to.equal(PRICE + PRICE * 5n);
    });

    it("all items faked -> caught on the first layer", async () => {
      const job = load("job_partial.json");        // generated with fraction = 1.0
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      const items = await market.getChallengedItems(id);
      await expect(market.connect(worker).revealLayer(id, 1, reveal(job, items, 1)))
        .to.emit(market, "JobSlashed");
    });

    it("only the final layer faked -> hidden layers pass, last layer slashed", async () => {
      const job = load("job_last-layer.json");
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      const items = await market.getChallengedItems(id);
      const L = job.layers.length;
      for (let l = 1; l < L; l++) {
        await expect(market.connect(worker).revealLayer(id, l, reveal(job, items, l)))
          .to.emit(market, "LayerVerified");
      }
      await expect(market.connect(worker).revealLayer(id, L, reveal(job, items, L)))
        .to.emit(market, "JobSlashed");
    });

    it("computing on the wrong inputs -> slashed", async () => {
      const job = load("job_swap-input.json");
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      const items = await market.getChallengedItems(id);
      await expect(market.connect(worker).revealLayer(id, 1, reveal(job, items, 1)))
        .to.emit(market, "JobSlashed");
    });

    it("tensor that is not in the committed tree -> rejected", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      const items = await market.getChallengedItems(id);
      const r = reveal(job, items, 2);
      const y = ethers.getBytes(r.ys[0]);
      y[0] = (y[0] + 1) % 128;
      r.ys = [ethers.hexlify(y), ...r.ys.slice(1)];
      await expect(market.connect(worker).revealLayer(id, 2, r)).to.be.revertedWith("bad output proof");
    });

    it("wrong model weights -> rejected", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      const items = await market.getChallengedItems(id);
      const r = reveal(job, items, 1);
      const W = ethers.getBytes(r.W);
      W[0] ^= 1;
      r.W = ethers.hexlify(W);
      await expect(market.connect(worker).revealLayer(id, 1, r)).to.be.revertedWith("bad weights proof");
    });
  });

  describe("timeouts", function () {
    it("worker that never commits is slashed", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      await createJob(market, renter, job);
      await market.connect(worker).acceptJob(0);
      await time.increase(3601);
      await expect(market.claimTimeout(0)).to.emit(market, "JobSlashed")
        .withArgs(0, worker.address, "no commit before deadline");
    });

    it("worker that stays silent after the challenge is slashed", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      const id = await runUntilChallenge(market, renter, worker, job);
      await time.increase(601);
      await expect(market.claimTimeout(id)).to.emit(market, "JobSlashed")
        .withArgs(id, worker.address, "no reveal before deadline");
      expect(await market.stakes(worker.address)).to.equal(STAKE - PRICE * 10n);
    });

    it("an expired seed block resets the challenge window", async () => {
      const job = load("job_none.json");
      const { market, renter, worker } = await deploy(job);
      await createJob(market, renter, job);
      await market.connect(worker).acceptJob(0);
      await market.connect(worker).commitResult(0, job.activationRoot, "");
      await mine(300);
      await expect(market.triggerChallenge(0)).to.emit(market, "ChallengeWindowReset");
      await mine(4);
      await expect(market.triggerChallenge(0)).to.emit(market, "ChallengeIssued");
    });
  });
});

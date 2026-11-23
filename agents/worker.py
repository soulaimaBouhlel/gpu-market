"""
WORKER agent (runs on VM 2 / VM 3).

Loop:
  1. watch for JobCreated events
  2. check the job uses the model we have, download + verify the inputs
  3. lock stake (acceptJob)
  4. run int8 inference, build the activation Merkle tree, upload activations
  5. commitResult(activationRoot)
  6. wait for the seed block, triggerChallenge
  7. reveal every layer for the challenged items -> paid (or slashed)

Cheat modes (for the evaluation, see agents/protocol.py):
  --cheat none | random | partial | last-layer | swap-input | silent | no-commit

Run:
    python3 agents/worker.py                       # honest
    python3 agents/worker.py --cheat partial --fraction 0.2
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.common import STATE_NAMES, Chain, get_json, put_json  # noqa: E402
from agents.protocol import CHEAT_MODES, JobCommitments, run_job  # noqa: E402
from crypto.commitments import build_input_tree, build_weights_tree  # noqa: E402
from ml.infer_int8 import load_model  # noqa: E402

STATE = {name: i for i, name in enumerate(STATE_NAMES)}


def log(msg):
    print(f"[worker] {msg}", flush=True)


class Worker:
    def __init__(self, cheat, fraction):
        self.chain = Chain()
        self.market = self.chain.market
        self.model = load_model()
        self.weights_root = build_weights_tree(self.model).root
        self.cheat = cheat
        self.fraction = fraction
        log(f"address {self.chain.address}  cheat mode: {cheat}")

    def ensure_stake(self, needed):
        free = self.market.functions.freeStake(self.chain.address).call()
        if free < needed:
            self.chain.send(self.market.functions.depositStake(), value=needed - free)
            log(f"deposited {self.chain.w3.from_wei(needed - free, 'ether')} ETH of stake")

    def handle(self, job_id):
        """Returns True if the job was taken by this worker."""
        m = self.market.functions
        job = m.getJob(job_id).call()
        if job.state != STATE["Open"]:
            return False

        # --- 1. only take jobs for a model we actually have
        root, dims, L = self.chain.registry.functions.getModel(job.modelId).call()
        if root != self.weights_root:
            log(f"job {job_id}: unknown model, skipping")
            return False

        # --- 2. download the inputs and check them against the renter's commitment
        inputs = get_json(job.inputsURI)["inputs"]
        if len(inputs) != job.numItems or build_input_tree(inputs).root != job.inputRoot:
            log(f"job {job_id}: inputs do not match inputRoot, skipping")
            return False

        # --- 3. lock stake
        self.ensure_stake(job.price * m.STAKE_MULTIPLIER().call())
        try:
            self.chain.send(m.acceptJob(job_id))
        except Exception as e:
            log(f"job {job_id}: could not accept ({e}), probably taken by another worker")
            return False
        log(f"job {job_id}: accepted ({job.numItems} items, price {self.chain.w3.from_wei(job.price, 'ether')} ETH)")

        if self.cheat == "no-commit":
            log(f"job {job_id}: [cheat] never committing")
            return True

        # --- 4. compute + commit
        t0 = time.time()
        traces = run_job(self.model, inputs, cheat=self.cheat, fraction=self.fraction, seed=job_id)
        c = JobCommitments(self.model, inputs, traces)
        url = put_json(f"jobs/{job_id}/activations.json", {"ys": [[s["y"] for s in t] for t in traces]})
        rcpt = self.chain.send(m.commitResult(job_id, c.activation_tree.root, url))
        seed_block = self.chain.events(rcpt, "ResultCommitted")[0]["args"]["seedBlock"]
        log(f"job {job_id}: computed in {time.time() - t0:.2f}s, committed root 0x{c.activation_tree.root.hex()[:16]}…")

        # --- 5. challenge (randomness comes from a block mined AFTER the commit)
        log(f"job {job_id}: waiting for seed block {seed_block}")
        self.chain.wait_block(seed_block)
        if m.getJob(job_id).call().state == STATE["Committed"]:
            self.chain.send(m.triggerChallenge(job_id))
        items = list(m.getChallengedItems(job_id).call())
        log(f"job {job_id}: challenged items {items}")

        if self.cheat == "silent":
            log(f"job {job_id}: [cheat] staying silent")
            return True

        # --- 6. reveal layer by layer
        for layer in range(1, L + 1):
            try:
                rcpt = self.chain.send(m.revealLayer(job_id, layer, c.layer_reveal(items, layer)))
            except Exception as e:
                log(f"job {job_id}: layer {layer} rejected ({e})")
                return True
            slashed = self.chain.events(rcpt, "JobSlashed")
            if slashed:
                log(f"job {job_id}: SLASHED at layer {layer}: {slashed[0]['args']['reason']} (gas {rcpt.gasUsed})")
                return True
            log(f"job {job_id}: layer {layer} verified on-chain (gas {rcpt.gasUsed})")

        state = STATE_NAMES[m.getJob(job_id).call().state]
        log(f"job {job_id}: final state {state}")
        if state == "Paid":
            self.chain.send(m.withdraw())
            log(f"job {job_id}: payment withdrawn")
        return True

    def run(self, once):
        next_block = 0
        log("listening for jobs…")
        while True:
            head = self.chain.w3.eth.block_number
            for ev in self.chain.logs("JobCreated", from_block=next_block):
                if self.handle(ev["args"]["jobId"]) and once:
                    return
            next_block = head + 1
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheat", choices=CHEAT_MODES, default="none")
    ap.add_argument("--fraction", type=float, default=0.3, help="share of faked items for --cheat partial")
    ap.add_argument("--once", action="store_true", help="exit after handling one job")
    args = ap.parse_args()
    Worker(args.cheat, args.fraction).run(args.once)


if __name__ == "__main__":
    main()

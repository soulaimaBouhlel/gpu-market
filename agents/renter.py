"""
RENTER agent (runs on VM 1).

  1. take N test images, quantize them, build the input Merkle tree
  2. upload the inputs to storage, createJob(inputRoot) + lock the payment
  3. follow the job; claim a timeout if the worker misses a deadline
  4. when paid: download the activations, check them against the
     on-chain activationRoot, print the predictions

Run:
    python3 agents/renter.py --items 10 --price 1
"""
import argparse
import os
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.common import STATE_NAMES, Chain, get_json, put_json  # noqa: E402
from crypto.commitments import build_activation_tree, build_input_tree  # noqa: E402
from ml.infer_int8 import ROOT, quantize_input  # noqa: E402

TERMINAL = {"Paid", "Slashed", "Cancelled"}


def log(msg):
    print(f"[renter] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0, help="first test image to use")
    ap.add_argument("--price", type=float, default=1.0, help="price in ETH")
    ap.add_argument("--compute-window", type=int, default=120, help="seconds for the worker to commit")
    ap.add_argument("--model-id", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900, help="give up following the job after N seconds")
    args = ap.parse_args()

    chain = Chain()
    m = chain.market.functions
    data = np.load(os.path.join(ROOT, "data", "float_model.npz"))
    X = data["X_test"][args.offset:args.offset + args.items]
    labels = data["y_test"][args.offset:args.offset + args.items]

    # --- 1-2. commit to the inputs and post the job
    inputs = [quantize_input(x) for x in X]
    input_root = build_input_tree(inputs).root
    url = put_json(f"inputs/{uuid.uuid4().hex}.json", {"inputs": inputs})
    price = chain.w3.to_wei(args.price, "ether")
    rcpt = chain.send(m.createJob(args.model_id, input_root, len(inputs), url, args.compute_window), value=price)
    job_id = chain.events(rcpt, "JobCreated")[0]["args"]["jobId"]
    log(f"job {job_id} created: {len(inputs)} items, {args.price} ETH locked in escrow")
    log(f"inputRoot 0x{input_root.hex()[:16]}…")

    # --- 3. follow the job
    last, t_start = None, time.time()
    while True:
        job = m.getJob(job_id).call()
        state = STATE_NAMES[job.state]
        if state != last:
            log(f"job {job_id}: {state}")
            last = state
        if state in TERMINAL:
            break
        now = chain.w3.eth.get_block("latest")["timestamp"]
        if state in ("Assigned", "Challenged") and now > job.deadline:
            log(f"job {job_id}: worker missed its deadline -> claimTimeout")
            chain.send(m.claimTimeout(job_id))
            continue
        if time.time() - t_start > args.timeout:
            log("giving up (timeout)")
            return
        time.sleep(2)

    # --- 4. results
    if state == "Paid":
        acts = get_json(job.resultsURI)["ys"]
        traces = [[{"y": y} for y in item] for item in acts]
        ok = build_activation_tree(traces).root == job.activationRoot
        log(f"downloaded activations match the on-chain root: {ok}")
        if ok:
            preds = [int(np.argmax(item[-1])) for item in acts]
            acc = float(np.mean(np.array(preds) == labels))
            log(f"predictions: {preds}")
            log(f"labels:      {labels.tolist()}")
            log(f"accuracy:    {acc:.2f}")
    elif state == "Slashed":
        reason = [e["args"]["reason"] for e in chain.logs("JobSlashed", 0) if e["args"]["jobId"] == job_id]
        log(f"worker was slashed: {reason[0] if reason else '?'}")

    bal = m.balances(chain.address).call()
    if bal > 0:
        chain.send(m.withdraw())
        log(f"withdrew {chain.w3.from_wei(bal, 'ether')} ETH (refund + compensation)")


if __name__ == "__main__":
    main()

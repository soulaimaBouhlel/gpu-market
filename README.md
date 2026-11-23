# Verifiable GPU Marketplace
 
**A blockchain marketplace for AI computation where the smart contract itself checks the work before paying.**
 
Renters pay to have a neural network run on their data. Workers with spare
computing power do the work. The renter never has to trust the worker: the
smart contract verifies the computation with cryptographic proofs, pays honest
workers automatically, and confiscates the stake of cheaters.
 
---
 
## The problem
 
Renting computing power from strangers is cheap, but how do you know the
worker actually did the work? A dishonest worker can return random results, use
a cheaper model, or compute on the wrong data, and still ask to be paid.
Recomputing everything yourself defeats the purpose of renting.
 
## The solution
 
```
 RENTER                         SMART CONTRACT                        WORKER
   │  createJob(inputRoot)           │                                   │
   │  + payment ───────────────────► │ escrow                            │
   │                                 │ ◄──────── acceptJob() + stake ────│
   │                                 │                                   │ runs the model,
   │                                 │ ◄──── commitResult(activationRoot)│ records every step
   │                                 │                                   │
   │                                 │ picks 5 random images from a      │
   │                                 │ block mined AFTER the commit      │
   │                                 │                                   │
   │                                 │ ◄────── revealLayer(1..3) ────────│
   │                                 │  ① Merkle proofs: data matches the commitments
   │                                 │  ② Freivalds: W·x + b = acc, checked in O(n²)
   │                                 │  ③ requantization: outputs follow the model
   │                                 │                                   │
   │ ◄── refund + compensation ──────┤ any check fails → SLASHED         │
   │                                 ├── all checks pass → PAID ────────►│
```
 
Four ideas make this work:
 
| Idea | What it guarantees |
|---|---|
| **Merkle commitments** | the worker commits to every intermediate result before knowing what will be checked, and can't change its story afterwards |
| **Freivalds' algorithm** | checking a matrix multiplication costs O(n²) instead of O(n³); a wrong result passes with probability ≈ 4 × 10⁻¹⁹ |
| **int8 quantization** | integer-only inference gives bit-identical results on every machine, so honest workers are never wrongly accused |
| **Staking and slashing** | a cheater loses 10× the job price when caught, so cheating has a negative expected gain |
 
No validators and no trusted server: **the contract is the judge.**
 
---
 
## Results
 
| Measure | Result |
|---|---|
| Model | MLP 64 → 32 → 16 → 10 on 8×8 handwritten digits |
| Accuracy (float → int8) | 96.2% → 96.4% |
| Determinism across 3 VMs | identical activation hash on every machine |
| Gas to verify one job (5 images, 3 layers) | 1.87 M + 0.70 M + 0.42 M ≈ **2.99 M** |
| Tests | 12 Python + 22 Solidity (incl. Python == Solidity cross-checks) |
 
### Attack scenarios (run on 3 VMs)
 
| Worker behavior | Outcome |
|---|---|
| honest | ✅ paid |
| returns random outputs | ❌ slashed: computation check failed |
| fakes 50% of the items | ❌ slashed: computation check failed |
| fakes only the last layer | ❌ layers 1–2 accepted, slashed at layer 3 |
| computes on the wrong inputs | ❌ slashed: computation check failed |
| stays silent after the challenge | ❌ slashed: no reveal before deadline |
| never commits | ❌ slashed: no commit before deadline |
 
### Partial cheating
 
A worker faking only a few items can get lucky. `scripts/detection_experiment.py`
simulates the contract's random challenge and compares it with the formula
`P(caught) = 1 − C(n−f, k) / C(n, k)`:
 
| Items | Faked | P(caught) formula | Simulated | Expected gain / price |
|---|---|---|---|---|
| 10 | 1 | 0.500 | 0.500 | −4.95 |
| 50 | 1 | 0.100 | 0.099 | −0.98 |
| 200 | 20 | 0.413 | 0.404 | −4.07 |
 
Even when detection is unlikely, the expected gain stays **negative**: the stake
lost when caught outweighs the work saved.
 
---
 
## Architecture
 
```
 ┌──────────────────── VM 1 "chain" 192.168.56.10 ─────────────────────┐
 │  Ganache :8545  ── ModelRegistry · JobMarket · LayerVerifier         │
 │  storage :8000  ── inputs, activations (checked against roots)       │
 │  renter.py                                                           │
 └──────────────▲──────────────────────────────────▲────────────────────┘
                │ JSON-RPC / HTTP                   │
   ┌────────────┴─────────────┐        ┌────────────┴─────────────┐
   │ VM 2 "worker-a" .11      │        │ VM 3 "worker-b" .12      │
   │ worker.py                │        │ worker.py                │
   └──────────────────────────┘        └──────────────────────────┘
```
 
| Layer | Technology |
|---|---|
| Blockchain | Ganache (local Ethereum), Solidity 0.8.24 |
| Contract tooling | Hardhat (compile, test, deploy) |
| Agents | Python 3, web3.py |
| Model | scikit-learn (training), NumPy (integer inference) |
| Infrastructure | 3 VirtualBox VMs (Ubuntu Desktop + 2× Ubuntu Server) |
 
---
 
## Project structure
 
```
gpu-market/
├── ml/            train.py · quantize.py · infer_int8.py   the int8 model
├── crypto/        encoding · merkle · freivalds · commitments   proof toolbox (Python)
├── contracts/     JobMarket · ModelRegistry · lib/LayerVerifier   on-chain verifier
├── agents/        renter.py · worker.py · protocol.py · common.py
├── storage/       server.py                                off-chain file storage
├── scripts/       deploy.js · register_model.py · gen_fixtures.py · detection_experiment.py
├── test/          Hardhat tests (Solidity)
├── tests_py/      pytest tests (Python)
└── e2e/           run_local.sh                             full-system test, 7 scenarios
```
 
---
 
## Quick start (one machine)
 
Requirements: Node.js 20+, Python 3.12+.
 
```bash
npm install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
 
python3 ml/train.py && python3 ml/quantize.py   # build the int8 model
python3 -m pytest -q                            # 12 Python tests
python3 scripts/gen_fixtures.py && npx hardhat test   # 22 contract tests
bash e2e/run_local.sh                           # Ganache + all 7 scenarios
```
 
The full step-by-step setup, including the 3-VM deployment, is in **[GUIDE.md](GUIDE.md)**.
 
---
 
## Limits and future work
 
- **Randomness** comes from a block hash: fine on a local chain, not on a public one.
  Production would use a VRF (e.g. Chainlink) or commit-reveal.
- **Gas grows with layer size.** On-chain verification suits small models, not LLMs.
  Future work: verify on-chain only when the renter disputes a result (optimistic model).
- **Partial cheating** on very few items can go undetected: the stake makes it
  unprofitable, not impossible.
- **AI fraud detection** (future work): learn from slashing events which jobs look
  suspicious, and check those more heavily.
- **Test ETH only**, and Ganache is no longer maintained (the Hardhat node is the alternative: `CHAIN=hardhat bash e2e/run_local.sh`).
## Related work
 
- **Verde** (Gensyn, 2025): refereed delegation for ML, dispute narrowed to one operator
- **opML** (ORA, 2024): optimistic ML with interactive fraud proofs
- **Slalom** (Tramèr & Boneh, 2019): Freivalds' algorithm for neural-network layers
- **Proof of Sampling** (Hyperbolic, 2024): game-theoretic sampling for decentralized inference
---
 
**Author:** Soulaima Bouhlel, software engineering student, ISIMS (Sfax, Tunisia)
 

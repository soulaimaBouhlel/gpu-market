#!/usr/bin/env bash
# Full system test on ONE machine: chain + storage + deploy + register,
# then one job per scenario (honest worker, then each cheat mode).
# Use it before going multi-VM, and as a regression test.
#
# Run from the project root:
#   bash e2e/run_local.sh            (all scenarios)
#   bash e2e/run_local.sh none       (only the honest scenario)
#   CHAIN=hardhat bash e2e/run_local.sh   (use the Hardhat node instead of Ganache)
set -euo pipefail
cd "$(dirname "$0")/.."

HH_CONFIG="${HH_CONFIG:-hardhat.config.js}"
CHAIN="${CHAIN:-ganache}"          # ganache (default, like in class) or hardhat
export RPC_URL=http://127.0.0.1:8545
export STORAGE_URL=http://127.0.0.1:8000
export DEPLOYMENTS=deployments/localhost.json
export REVEAL_WINDOW=20            # short reveal window so the "silent" scenario is fast
DEPLOYER=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
RENTER=0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d
WORKER=0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a
LOGS=e2e/logs; mkdir -p "$LOGS"

cleanup() {
  pkill -f "ganache" 2>/dev/null || true
  pkill -f "hardhat.* node" 2>/dev/null || true
  pkill -f "storage/server.py" 2>/dev/null || true
}
trap cleanup EXIT

if curl -s -o /dev/null http://127.0.0.1:8545; then
  echo "Port 8545 is already in use: stop the running Hardhat node first."; exit 1
fi

echo "== starting blockchain node ($CHAIN) and storage"
if [ "$CHAIN" = "ganache" ]; then
  npm run chain > "$LOGS/node.log" 2>&1 &
  NETWORK=ganache
else
  npx hardhat --config "$HH_CONFIG" node > "$LOGS/node.log" 2>&1 &
  NETWORK=localhost
fi
python3 storage/server.py > "$LOGS/storage.log" 2>&1 &
for i in $(seq 1 60); do
  curl -s -X POST -H 'Content-Type: application/json' \
    --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' $RPC_URL >/dev/null 2>&1 && break
  sleep 1
done

echo "== deploying contracts"
npx hardhat --config "$HH_CONFIG" run scripts/deploy.js --network $NETWORK
echo "== registering the model"
PRIVATE_KEY=$DEPLOYER python3 scripts/register_model.py

SCENARIOS=("${@:-none random partial last-layer swap-input silent no-commit}")
for mode in ${SCENARIOS[@]}; do
  echo
  echo "=================== scenario: $mode ==================="
  PRIVATE_KEY=$WORKER python3 agents/worker.py --cheat "$mode" --fraction 0.5 --once > "$LOGS/worker_$mode.log" 2>&1 &
  WPID=$!
  COMPUTE_WINDOW=60; [ "$mode" = "no-commit" ] && COMPUTE_WINDOW=15
  PRIVATE_KEY=$RENTER python3 agents/renter.py --items 10 --price 1 --compute-window $COMPUTE_WINDOW --timeout 300
  wait $WPID || true
  echo "---- worker log"
  cat "$LOGS/worker_$mode.log"
done

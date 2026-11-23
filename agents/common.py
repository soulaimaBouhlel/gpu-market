"""
Shared plumbing for the agents: configuration, web3 connection, signed
transactions, event polling and the storage client.

Configuration comes from environment variables (or a .env file):
    RPC_URL        http://<VM1-IP>:8545
    STORAGE_URL    http://<VM1-IP>:8000
    PRIVATE_KEY    the agent's wallet key (one per VM / role)
    DEPLOYMENTS    path to deployments/localhost.json (copied from VM 1)
"""
import json
import os
import time
import urllib.request

from web3 import Web3
from web3.logs import DISCARD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=os.path.join(ROOT, ".env")):
    """Tiny .env loader (KEY=VALUE lines). Real env vars take precedence."""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


class Chain:
    def __init__(self, private_key=None):
        load_env()
        self.w3 = Web3(Web3.HTTPProvider(os.environ.get("RPC_URL", "http://127.0.0.1:8545")))
        if not self.w3.is_connected():
            raise SystemExit("cannot reach the blockchain node: check RPC_URL")
        self.account = self.w3.eth.account.from_key(private_key or os.environ["PRIVATE_KEY"])
        self.address = self.account.address

        dep_path = os.environ.get("DEPLOYMENTS", os.path.join(ROOT, "deployments", "localhost.json"))
        with open(dep_path) as f:
            dep = json.load(f)
        self.registry = self.w3.eth.contract(address=dep["ModelRegistry"]["address"], abi=dep["ModelRegistry"]["abi"], decode_tuples=True)
        self.market = self.w3.eth.contract(address=dep["JobMarket"]["address"], abi=dep["JobMarket"]["abi"], decode_tuples=True)

    def send(self, fn, value=0):
        """Build, sign locally with this agent's key, send, wait. Raises on revert."""
        tx = fn.build_transaction({
            "from": self.address,
            "value": value,
            "nonce": self.w3.eth.get_transaction_count(self.address, "pending"),
            "chainId": self.w3.eth.chain_id,
        })
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt.status != 1:
            raise RuntimeError(f"transaction reverted: {tx_hash.hex()}")
        return receipt

    def events(self, receipt, name):
        return getattr(self.market.events, name)().process_receipt(receipt, errors=DISCARD)

    def logs(self, name, from_block):
        return getattr(self.market.events, name)().get_logs(from_block=from_block)

    def wait_block(self, target):
        while self.w3.eth.block_number <= target:
            time.sleep(1)


# ---------------------------------------------------------------- storage
def storage_url():
    load_env()
    return os.environ.get("STORAGE_URL", "http://127.0.0.1:8000").rstrip("/")


def put_json(path, obj):
    url = f"{storage_url()}/{path}"
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), method="PUT")
    urllib.request.urlopen(req).close()
    return url


def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


STATE_NAMES = ["None", "Open", "Assigned", "Committed", "Challenged", "Paid", "Slashed", "Cancelled"]

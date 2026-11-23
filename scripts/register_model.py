"""
Register the int8 model on-chain: only its weights Merkle root + dimensions.

Run once on VM 1 after deploying the contracts:
    python3 scripts/register_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.common import Chain  # noqa: E402
from agents.protocol import model_dims  # noqa: E402
from crypto.commitments import build_weights_tree  # noqa: E402
from ml.infer_int8 import load_model  # noqa: E402


def main():
    chain = Chain()
    model = load_model()
    root = build_weights_tree(model).root
    dims = model_dims(model)
    receipt = chain.send(chain.registry.functions.registerModel(root, dims))
    ev = chain.registry.events.ModelRegistered().process_receipt(receipt)[0]["args"]
    print(f"[register] modelId={ev['modelId']} dims={dims}")
    print(f"[register] weightsRoot=0x{root.hex()}")


if __name__ == "__main__":
    main()


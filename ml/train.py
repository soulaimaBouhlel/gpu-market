"""
Step 1a — Train a small float MLP on the scikit-learn "digits" dataset.

Dataset: 1797 images of handwritten digits, 8x8 pixels, values 0..16.
It ships inside scikit-learn, so no download is needed.

Architecture: 64 -> 32 -> 16 -> 10  (ReLU hidden layers)

Output: data/float_model.npz  (float weights + biases + test split)

Run from the project root:
    python3 ml/train.py
"""
import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "float_model.npz")


def load_data():
    digits = load_digits()
    X = digits.data / 16.0           # floats in [0, 1]
    y = digits.target
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def main():
    X_train, X_test, y_train, y_test = load_data()

    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=600,
        random_state=0,
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"[train] float accuracy on test set: {acc:.4f}")

    # sklearn stores weights as (n_in, n_out); we use (n_out, n_in) so that y = W @ x
    weights = [w.T.astype(np.float64) for w in clf.coefs_]
    biases = [b.astype(np.float64) for b in clf.intercepts_]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT,
        **{f"W{i}": w for i, w in enumerate(weights)},
        **{f"b{i}": b for i, b in enumerate(biases)},
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        num_layers=len(weights),
    )
    print(f"[train] saved {OUT}")


if __name__ == "__main__":
    main()

"""
Off-chain storage: a minimal HTTP file server (runs on VM 1).

    PUT /<path>   store the request body under storage/files/<path>
    GET /<path>   return it

Big data (inputs, activations, results) lives here; only hashes go on-chain.
Integrity does not depend on trusting this server: every file is checked
against a Merkle root stored in the smart contract.

Run:
    python3 storage/server.py            (listens on 0.0.0.0:8000)
"""
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
PORT = int(os.environ.get("STORAGE_PORT", "8000"))


def safe_path(url_path):
    rel = os.path.normpath(url_path.lstrip("/"))
    if rel.startswith("..") or os.path.isabs(rel):
        raise ValueError("bad path")
    return os.path.join(BASE, rel)


class Handler(BaseHTTPRequestHandler):
    def do_PUT(self):
        try:
            path = safe_path(self.path)
        except ValueError:
            return self.send_error(400, "bad path")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(body)
        self.send_response(201)
        self.end_headers()

    def do_GET(self):
        try:
            path = safe_path(self.path)
        except ValueError:
            return self.send_error(400, "bad path")
        if not os.path.isfile(path):
            return self.send_error(404, "not found")
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[storage] {self.command} {self.path}")


if __name__ == "__main__":
    os.makedirs(BASE, exist_ok=True)
    print(f"[storage] serving {BASE} on 0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

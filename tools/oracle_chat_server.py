#!/usr/bin/env python3
import json
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLE_BIN = ROOT / "build" / "oracle"


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ["/", "/chat"]:
            html = (ROOT / "ui" / "chat.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return

        size = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(size) or b"{}")
        message = str(data.get("message", ""))
        steps = str(data.get("steps", 200))
        dt = str(data.get("dt", 1e-4))
        diffusion = str(data.get("diffusion", "explicit"))

        if not ORACLE_BIN.exists():
            self._send_json({"error": "Build oracle first: cmake -S . -B build && cmake --build build -j"}, status=500)
            return

        cmd = [
            str(ORACLE_BIN),
            steps,
            dt,
            "--diffusion",
            diffusion,
            "--prompt",
            message,
            "--json",
        ]

        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            self._send_json({"error": proc.stderr or proc.stdout}, status=500)
            return

        self._send_json(json.loads(proc.stdout))


if __name__ == "__main__":
    print("Serving Oracle chat on http://0.0.0.0:8080")
    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()

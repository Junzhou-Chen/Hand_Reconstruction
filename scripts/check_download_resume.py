#!/usr/bin/env python3
"""Remote-only small HTTP checks for downloader resume and integrity behavior."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from download_twm_data import fetch_file

PAYLOAD = b"handloop-download-check" * 1024


class Handler(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        offset = int(self.headers.get("Range", "bytes=0-")[6:-1])
        self.requests.append(offset)
        ignore = self.path.startswith("/ignore")
        self.send_response(200 if ignore or not offset else 206)
        if offset and not ignore:
            self.send_header("Content-Range", f"bytes {offset}-{len(PAYLOAD)-1}/{len(PAYLOAD)}")
        self.end_headers()
        self.wfile.write(PAYLOAD if ignore else PAYLOAD[offset:])

    def log_message(self, *args):
        pass


class ResumeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / "test.tar"
        self.entry = {"path": "test.tar", "size": len(PAYLOAD), "hash_kind": "sha256",
                      "hash": hashlib.sha256(PAYLOAD).hexdigest(),
                      "url": f"http://127.0.0.1:{self.server.server_port}/range"}
        Handler.requests.clear()

    def test_resume_and_skip_verified(self):
        self.target.with_name("test.tar.partial").write_bytes(PAYLOAD[:1234])
        fetch_file(self.entry, self.target)
        self.assertEqual(Handler.requests, [1234])
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        fetch_file(self.entry, self.target)
        self.assertEqual(Handler.requests, [1234])

    def test_server_ignoring_range(self):
        self.entry["url"] = self.entry["url"].replace("/range", "/ignore")
        self.target.with_name("test.tar.partial").write_bytes(PAYLOAD[:1234])
        fetch_file(self.entry, self.target)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)

    def test_checksum_rejects_and_quarantines_bad_partial(self):
        self.target.with_name("test.tar.partial").write_bytes(b"x" * len(PAYLOAD))
        with patch("download_twm_data.time.sleep"):
            fetch_file(self.entry, self.target, attempts=2)
        self.assertEqual(self.target.read_bytes(), PAYLOAD)
        self.assertEqual(len(list(self.target.parent.glob("*.corrupt-*"))), 1)

    def test_wrong_source_hash_never_gets_final_name(self):
        self.entry["hash"] = "0" * 64
        with self.assertRaises(RuntimeError):
            fetch_file(self.entry, self.target, attempts=1)
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()

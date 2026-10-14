#!/usr/bin/env python3
"""Small downloader checks; execute on ds/its Pod, never on the local workstation.

Creates synthetic ZIP and TAR.GZ fixtures in a temporary remote directory. The
HTTP fixture serves localhost only and never contacts a dataset service.
"""
from contextlib import contextmanager
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import re
import tarfile
import tempfile
import threading
import unittest
import zipfile

from download_comparison_data import (
    AuthenticationRequired,
    fetch_arctic,
    validate_gzip_tar,
)


def zip_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture.txt", b"Synthetic downloader fixture.\n" * 16)
        archive.writestr("nested/metadata.json", b'{"synthetic": true}\n')
    return buffer.getvalue()


def tar_gzip_bytes():
    buffer = io.BytesIO()
    content = b"Synthetic TAR.GZ fixture.\n" * 16
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("fixture.txt")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@contextmanager
def serve_zip(payload, mode):
    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Fixtures must not print even the dummy authentication body.

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            request_range = self.headers.get("Range")
            requests_seen.append(request_range)
            if mode == "html":
                body = b"<html><body>Sign in</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=UTF-8")
            elif mode == "ignore_range" or request_range is None:
                body = payload
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
            else:
                match = re.fullmatch(r"bytes=(\d+)-", request_range)
                if match is None:
                    self.send_error(400)
                    return
                offset = int(match[1])
                if offset >= len(payload):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(payload)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = payload[offset:]
                self.send_response(206)
                self.send_header("Content-Type", "application/zip")
                self.send_header(
                    "Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
                )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/fixture.zip", requests_seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ArcticDownloadChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="comparison-download-check-")
        self.addCleanup(self.temporary.cleanup)
        self.target = Path(self.temporary.name) / "fixture.zip"
        self.partial = self.target.with_name(self.target.name + ".partial")
        self.payload = zip_bytes()
        self.expected_hash = hashlib.sha256(self.payload).hexdigest()
        self.updates = []

    def fetch(self, url):
        return fetch_arctic(
            {"url": url, "sha256": self.expected_hash},
            self.target,
            {"username": "dummy", "password": "dummy"},
            lambda **changes: self.updates.append(changes),
            attempts=1,
        )

    def assert_completed(self, result):
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse(self.partial.exists())
        self.assertEqual(result["sha256"], self.expected_hash)
        self.assertEqual(result["size"], len(self.payload))

    def test_partial_resumes_with_valid_content_range(self):
        offset = 37
        self.partial.write_bytes(self.payload[:offset])
        with serve_zip(self.payload, "range") as (url, seen):
            result = self.fetch(url)
        self.assertEqual(seen, [f"bytes={offset}-"])
        self.assert_completed(result)

    def test_ignored_range_replaces_partial_instead_of_appending(self):
        offset = 53
        self.partial.write_bytes(self.payload[:offset])
        with serve_zip(self.payload, "ignore_range") as (url, seen):
            result = self.fetch(url)
        self.assertEqual(seen, [f"bytes={offset}-"])
        self.assert_completed(result)

    def test_login_html_is_rejected_even_with_http_200(self):
        with serve_zip(self.payload, "html") as (url, seen):
            with self.assertRaises(AuthenticationRequired):
                self.fetch(url)
        self.assertEqual(len(seen), 1)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.partial.exists())

    def test_complete_partial_is_verified_and_renamed_without_network(self):
        self.partial.write_bytes(self.payload)
        with serve_zip(self.payload, "range") as (url, seen):
            result = self.fetch(url)
        self.assertEqual(seen, [])
        self.assert_completed(result)


class GzipTarChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="comparison-gzip-check-")
        self.addCleanup(self.temporary.cleanup)
        self.target = Path(self.temporary.name) / "fixture.tar.gz"
        self.payload = tar_gzip_bytes()

    def test_complete_archive_passes(self):
        self.target.write_bytes(self.payload)
        validate_gzip_tar(self.target, len(self.payload))

    def test_truncated_archive_fails_even_if_expected_size_matches(self):
        truncated = self.payload[:-5]
        self.target.write_bytes(truncated)
        with self.assertRaises((EOFError, OSError, ValueError)):
            validate_gzip_tar(self.target, len(truncated))

    def test_corrupted_crc_fails_even_if_expected_size_matches(self):
        corrupted = bytearray(self.payload)
        corrupted[-8] ^= 1  # Change only the gzip CRC, preserving compressed data.
        self.target.write_bytes(corrupted)
        with self.assertRaises((gzip.BadGzipFile, OSError, ValueError)):
            validate_gzip_tar(self.target, len(corrupted))


if __name__ == "__main__":
    if not Path("/sciclone/home/jchen57").is_dir():
        raise SystemExit("Run this check on ds or its Pod; local execution is disabled.")
    unittest.main(verbosity=2)

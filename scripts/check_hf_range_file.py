"""Run in the remote Pod: python hf_range_file_selftest.py /path/to/scripts."""
from __future__ import annotations

import io
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

import h5py
import numpy as np

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from hf_range_file import HFConcatRangeFile, HTTPRangeError


class Body:
    def __init__(self, data):
        self.data = io.BytesIO(data)
        self.read_calls = 0

    def read(self, count, decode_content=False):
        assert decode_content is False
        self.read_calls += 1
        return self.data.read(count)


class Response:
    def __init__(self, path, data, start, end, fault=None):
        self.status_code = 200 if fault == "200" else 206
        self.history = []
        self.url = "https://signed.example/" + path + "?token=DO_NOT_PRINT"
        range_start = start + (fault == "range")
        total = len(data) + (fault == "total")
        payload = data[start:end + 1]
        self.headers = {
            "Content-Range": f"bytes {range_start}-{end}/{total}",
            "Content-Length": str(len(payload) + (fault == "length")),
            "Content-Encoding": "gzip" if fault == "encoding" else "identity",
        }
        self.raw = Body(payload[:-1] if fault == "short" else payload)
        self.closed = False

    def close(self):
        self.closed = True


class Session:
    def __init__(self, files, fault=None):
        self.files = files
        self.fault = fault
        self.responses = []
        self.closed = False

    def get(self, url, *, headers, timeout, stream, allow_redirects):
        assert stream is True and allow_redirects is True
        assert headers["Accept-Encoding"] == "identity"
        path = unquote(urlparse(url).path).split("/")[-1]
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"])
        assert match
        start, end = map(int, match.groups())
        assert 0 <= start <= end < len(self.files[path])
        response = Response(path, self.files[path], start, end, self.fault)
        self.responses.append(response)
        return response

    def close(self):
        self.closed = True


def remote_view(data, cuts, *, base=0, size=None, fault=None, cache_blocks=2):
    boundaries = [0, *cuts, len(data)]
    files = {
        f"fixture.part_{i}": data[a:b]
        for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]))
    }
    parts = [{"path": key, "size": len(value)} for key, value in files.items()]
    session = Session(files, fault)
    remote = HFConcatRangeFile(
        "fixture/synthetic", "1" * 40, parts,
        base=base, size=size, block_size=64, cache_blocks=cache_blocks,
        max_read_size=64 * 1024, retries=1, session=session,
    )
    return remote, session


def must_raise(kind, callback):
    try:
        callback()
    except kind as exc:
        assert "DO_NOT_PRINT" not in str(exc)
    else:
        raise AssertionError(f"expected {kind.__name__}")


def main():
    payload = bytes(range(251)) * 8
    # Cross two part boundaries within one base-shifted cached block.
    remote, session = remote_view(payload, [81, 99, 117, 1001], base=53, size=1600)
    assert remote.readable() and remote.seekable() and not remote.writable()
    assert remote.read(100) == payload[53:153]
    remote.seek(17)
    target = bytearray(93)
    assert remote.readinto(target) == len(target)
    assert bytes(target) == payload[70:163]
    remote.seek(10, io.SEEK_CUR)
    assert remote.tell() == 120
    remote.seek(-9, io.SEEK_END)
    assert remote.read(200) == payload[1644:1653]
    assert remote.read(1) == b"" and remote.readinto(bytearray(3)) == 0
    remote.seek(99, io.SEEK_END)
    assert remote.read() == b""
    must_raise(ValueError, lambda: remote.seek(-1))
    must_raise(ValueError, lambda: remote.seek(0, 99))
    remote.seek(0)
    before = remote.network_requests
    remote.read(64)
    remote.seek(0)
    remote.read(64)
    assert remote.network_requests <= before + 4
    assert remote.stats["cache_hits"] >= 1
    assert remote.stats["cached_blocks"] <= 2
    assert remote.stats["cached_bytes"] <= 128
    assert all(response.closed for response in session.responses)
    remote.close()
    assert not session.closed
    must_raise(ValueError, remote.tell)

    # A huge logical read fails without any network transfer.
    remote, session = remote_view(payload, [1000])
    remote.max_read_size = 100
    must_raise(ValueError, remote.readall)
    must_raise(ValueError, lambda: remote.readinto(bytearray(101)))
    assert not session.responses
    remote.close()

    for fault in ["200", "range", "total", "length", "encoding", "short"]:
        remote, session = remote_view(payload, [1000], fault=fault)
        must_raise(HTTPRangeError, lambda: remote.read(5))
        assert len(session.responses) == 1
        assert session.responses[0].closed
        if fault != "short":
            assert session.responses[0].raw.read_calls == 0
        remote.close()

    # A real, small HDF5 file exercises its fileobj driver over fake ranges.
    fixture = io.BytesIO()
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    with h5py.File(fixture, "w") as handle:
        handle.attrs["fixture"] = "synthetic"
        handle.create_dataset("trajectory/xyz", data=expected)
    hdf_bytes = fixture.getvalue()
    assert len(hdf_bytes) < 16 * 1024
    wrapped = b"t" * 512 + hdf_bytes + b"f" * 17
    remote, session = remote_view(
        wrapped, [137, 519, 777, 1234], base=512, size=len(hdf_bytes), cache_blocks=256,
    )
    with h5py.File(remote, "r") as handle:
        assert handle.attrs["fixture"] == "synthetic"
        np.testing.assert_array_equal(handle["trajectory/xyz"][:], expected)
        np.testing.assert_array_equal(handle["trajectory/xyz"][1:3, 1:], expected[1:3, 1:])
    stats = remote.stats
    remote.close()
    assert all(response.closed for response in session.responses)
    print({"status": "passed", "tests": "cross-part/window/seek/readinto/EOF/LRU/strict-HTTP/h5py", "fixture_bytes": len(hdf_bytes), "hdf_range_stats": stats})


if __name__ == "__main__":
    main()

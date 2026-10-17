"""Bounded HTTP Range access to immutable, concatenated Hugging Face files.

The caller supplies parts in concatenation order. ``base`` and ``size`` expose
an optional window, for example the HDF5 member after a 512-byte tar header.
No part is downloaded in full. Suitable for ``h5py.File(fileobj, "r")``.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
import io
import operator
import re
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import requests


class HTTPRangeError(OSError):
    """A remote range could not be read and validated."""


class HFConcatRangeFile(io.RawIOBase):
    """Read-only seekable view of ordered HF repository parts.

    ``revision`` must be an immutable 40-character commit SHA. Each part is a
    mapping containing ``path`` and its exact positive ``size`` in bytes.
    ``read(-1)`` and large individual reads fail before allocating more than
    ``max_read_size``. The LRU holds at most ``cache_blocks * block_size`` bytes.

    Network counters measure all attempted range GETs (including redirects)
    and response body bytes read, including a failed attempt. Redirect URLs,
    which can contain signed credentials, remain in memory and never appear
    in errors or statistics. A caller-supplied session remains caller-owned.
    """

    def __init__(
        self,
        repo_id: str,
        revision: str,
        parts: Sequence[Mapping[str, Any]],
        *,
        base: int = 0,
        size: int | None = None,
        block_size: int = 256 * 1024,
        cache_blocks: int = 256,
        max_read_size: int = 64 * 1024 * 1024,
        retries: int = 4,
        timeout: tuple[float, float] = (15, 60),
        session: requests.Session | None = None,
        repo_type: str = "dataset",
    ) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._redirects: dict[int, tuple[str, float]] = {}
        self._session = session
        self._owns_session = session is None
        self.network_bytes = 0
        self.network_requests = 0
        self.range_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id):
            raise ValueError("repo_id must be an owner/repository name")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("revision must be an immutable 40-character SHA")
        if repo_type not in {"dataset", "model", "space"}:
            raise ValueError("repo_type must be dataset, model, or space")
        self.repo_id = repo_id
        self.revision = revision
        self.block_size = operator.index(block_size)
        self.cache_blocks = operator.index(cache_blocks)
        self.max_read_size = operator.index(max_read_size)
        self.retries = operator.index(retries)
        if min(self.block_size, self.cache_blocks, self.max_read_size) <= 0:
            raise ValueError("block and read limits must be positive")
        if self.block_size > self.max_read_size:
            raise ValueError("block_size must not exceed max_read_size")
        if not 1 <= self.retries <= 10:
            raise ValueError("retries must be between 1 and 10")
        self.timeout = timeout
        prefix = {"dataset": "datasets/", "model": "", "space": "spaces/"}[repo_type]
        self._repo_url = "https://huggingface.co/" + prefix + quote(repo_id, safe="/")
        self._parts: list[dict[str, Any]] = []
        self._starts: list[int] = []
        total = 0
        for part in parts:
            path = str(part["path"])
            part_size = operator.index(part["size"])
            if (
                not path
                or path.startswith("/")
                or "\\" in path
                or any(p in {"", ".", ".."} for p in path.split("/"))
                or part_size <= 0
            ):
                raise ValueError("parts need relative paths and positive byte sizes")
            self._starts.append(total)
            self._parts.append({"path": path, "size": part_size})
            total += part_size
        if not self._parts:
            raise ValueError("at least one part is required")
        self.base = operator.index(base)
        self.size = total - self.base if size is None else operator.index(size)
        if self.base < 0 or self.size < 0 or self.base + self.size > total:
            raise ValueError("the requested window exceeds the concatenated parts")
        self.total_size = total
        self._position = 0
        if self._session is None:
            self._session = requests.Session()

    def readable(self) -> bool:
        self._checkClosed()
        return True

    def writable(self) -> bool:
        self._checkClosed()
        return False

    def seekable(self) -> bool:
        self._checkClosed()
        return True

    def tell(self) -> int:
        with self._lock:
            self._checkClosed()
            return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        with self._lock:
            self._checkClosed()
            offset = operator.index(offset)
            if whence == io.SEEK_SET:
                position = offset
            elif whence == io.SEEK_CUR:
                position = self._position + offset
            elif whence == io.SEEK_END:
                position = self.size + offset
            else:
                raise ValueError("invalid whence")
            if position < 0:
                raise ValueError("negative seek position")
            self._position = position
            return position

    def read(self, size: int = -1) -> bytes:
        with self._lock:
            self._checkClosed()
            size = -1 if size is None else operator.index(size)
            if size < -1:
                raise ValueError("invalid read size")
            remaining = max(0, self.size - self._position)
            requested = remaining if size == -1 else min(size, remaining)
            if requested > self.max_read_size:
                raise ValueError("read exceeds max_read_size; read in bounded chunks")
            if requested == 0:
                return b""
            chunks: list[bytes] = []
            position = self._position
            left = requested
            while left:
                block_index, block_offset = divmod(position, self.block_size)
                block = self._get_block(block_index)
                count = min(left, len(block) - block_offset)
                if count <= 0:
                    raise HTTPRangeError("unexpected end of cached range")
                chunks.append(block[block_offset : block_offset + count])
                position += count
                left -= count
            self._position = position
            return b"".join(chunks)

    def readinto(self, buffer: Any) -> int:
        with self._lock:
            self._checkClosed()
            view = memoryview(buffer)
            if view.readonly:
                raise TypeError("readinto needs a writable buffer")
            view = view.cast("B")
            if len(view) > self.max_read_size:
                raise ValueError("readinto exceeds max_read_size")
            data = self.read(len(view))
            view[: len(data)] = data
            return len(data)

    def readall(self) -> bytes:
        return self.read(-1)

    def _get_block(self, block_index: int) -> bytes:
        cached = self._cache.get(block_index)
        if cached is not None:
            self.cache_hits += 1
            self._cache.move_to_end(block_index)
            return cached
        self.cache_misses += 1
        start = block_index * self.block_size
        length = min(self.block_size, self.size - start)
        data = self._read_concat(self.base + start, length)
        self._cache[block_index] = data
        self._cache.move_to_end(block_index)
        while len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return data

    def _read_concat(self, start: int, length: int) -> bytes:
        chunks: list[bytes] = []
        while length:
            part_index = bisect.bisect_right(self._starts, start) - 1
            part_offset = start - self._starts[part_index]
            count = min(length, self._parts[part_index]["size"] - part_offset)
            chunks.append(self._read_part(part_index, part_offset, count))
            start += count
            length -= count
        return b"".join(chunks)

    def _read_part(self, part_index: int, start: int, length: int) -> bytes:
        part = self._parts[part_index]
        end = start + length - 1
        official = (
            f"{self._repo_url}/resolve/{self.revision}/{quote(part['path'], safe='/')}"
            f"?download=true&handloop_range={start}-{end}"
        )
        headers = {
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "HandLoop-HFRangeFile/1.0",
        }
        reason = "request failed"
        for attempt in range(self.retries):
            response = None
            cached = self._redirects.get(part_index)
            url = cached[0] if cached and time.monotonic() - cached[1] < 120 else official
            try:
                self.range_requests += 1
                self.network_requests += 1
                response = self._session.get(
                    url, headers=headers, timeout=self.timeout, stream=True,
                    allow_redirects=True,
                )
                self.network_requests += len(response.history)
                if response.status_code != 206:
                    reason = f"HTTP {response.status_code}, expected 206"
                    raise HTTPRangeError(reason)
                encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
                if encoding not in {"", "identity"}:
                    reason = "response is not identity encoded"
                    raise HTTPRangeError(reason)
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)",
                    response.headers.get("Content-Range", "").strip(),
                )
                if not match or tuple(map(int, match.groups())) != (start, end, part["size"]):
                    reason = "Content-Range does not match the requested immutable part"
                    raise HTTPRangeError(reason)
                content_length = response.headers.get("Content-Length")
                if content_length is not None and content_length != str(length):
                    reason = "Content-Length does not match the requested range"
                    raise HTTPRangeError(reason)
                # Headers are checked before reading: a server returning a whole
                # multi-GB file with status 200 cannot trigger a full download.
                body = response.raw.read(length + 1, decode_content=False)
                self.network_bytes += len(body)
                if len(body) != length:
                    reason = "response body length does not match the requested range"
                    raise HTTPRangeError(reason)
                if response.url != official:
                    self._redirects[part_index] = (response.url, time.monotonic())
                return body
            except HTTPRangeError:
                self._redirects.pop(part_index, None)
            except Exception as exc:
                # requests/urllib3 errors can embed a signed URL. Report only
                # their class, retaining neither exception text nor its cause.
                reason = f"transport error ({type(exc).__name__})"
                self._redirects.pop(part_index, None)
            finally:
                if response is not None:
                    response.close()
            if attempt + 1 < self.retries:
                time.sleep(min(0.5 * (2 ** attempt), 4.0))
        raise HTTPRangeError(
            f"part {part_index} bytes {start}-{end}: {reason} "
            f"after {self.retries} attempts"
        ) from None

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "network_bytes": self.network_bytes,
                "network_requests": self.network_requests,
                "range_requests": self.range_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cached_blocks": len(self._cache),
                "cached_bytes": sum(map(len, self._cache.values())),
            }

    def close(self) -> None:
        with self._lock:
            if not self.closed:
                self._cache.clear()
                self._redirects.clear()
                if self._owns_session and self._session is not None:
                    self._session.close()
                super().close()


HFRangeFile = HFConcatRangeFile

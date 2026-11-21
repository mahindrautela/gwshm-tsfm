"""Read single members of a large remote zip archive with HTTP range requests.

The OGW temperature dataset (figshare) ships as zip archives of 27-54 GB, of which we need
only one file per measurement. ``RemoteZip`` opens the archive over HTTP and transfers just
the central directory and the members that are read. Standard library only.

    with RemoteZip(url) as z:
        names = [n for n in z.namelist() if n.endswith("pc_f40kHz.h5")]
        data = z.read(names[0])
"""

from __future__ import annotations

import io
import time
import urllib.request
import zipfile


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP: every read is one range request.

    figshare redirects to a signed storage URL that expires after a while; on a failed
    request the redirect is resolved again from the original URL.
    """

    def __init__(self, url: str, retries: int = 6):
        self.origin, self.retries, self.pos = url, retries, 0
        self._resolve()

    def _resolve(self):
        req = urllib.request.Request(self.origin, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(3 + 5 * attempt)
                try:
                    self._resolve()                     # fresh signed URL
                except Exception:
                    pass
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


class RemoteZip(zipfile.ZipFile):
    """A zipfile.ZipFile whose bytes come from a URL (read-only)."""

    def __init__(self, url: str, buffer: int = 8 << 20):
        super().__init__(io.BufferedReader(HTTPRangeFile(url), buffer_size=buffer))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only sysfs access, replaceable by a recorded copy for tests.

Every pciescope module that reads ``/sys`` goes through ``Sysfs`` so the exact
same code runs against a fixture tree captured from the real box
(``RecordingSysfs`` writes the files a run reads into a directory that ``Sysfs``
can then be pointed at).  Reads use ``os.open``/``os.read`` (about 5 us a file
on sysfs) and never raise: a missing or unreadable attribute is ``None``.
"""

from __future__ import annotations

import os

DEFAULT_ROOT = "/sys"
_MAX_READ = 1 << 16


class Sysfs:
    """Read-only view of a sysfs tree rooted at ``root`` (default ``/sys``)."""

    def __init__(self, root: str | os.PathLike[str] = DEFAULT_ROOT) -> None:
        self.root = os.fspath(root).rstrip("/") or "/"

    def abspath(self, rel: str) -> str:
        return f"{self.root}/{rel.lstrip('/')}"

    def read_bytes(self, rel: str, limit: int = _MAX_READ) -> bytes | None:
        """Raw bytes of an attribute, None if missing or unreadable."""
        try:
            fd = os.open(self.abspath(rel), os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            return None
        try:
            return os.read(fd, limit)
        except OSError:
            return None
        finally:
            os.close(fd)

    def read(self, rel: str) -> str | None:
        """Stripped text of an attribute, None if missing or unreadable."""
        raw = self.read_bytes(rel)
        return None if raw is None else raw.decode("utf-8", "replace").strip()

    def read_int(self, rel: str, base: int = 10) -> int | None:
        """Integer attribute (``base=0`` accepts ``0x`` prefixes), else None."""
        txt = self.read(rel)
        if not txt:
            return None
        try:
            return int(txt.split()[0], base)
        except ValueError:
            return None

    def exists(self, rel: str) -> bool:
        return os.path.lexists(self.abspath(rel))

    def isdir(self, rel: str) -> bool:
        return os.path.isdir(self.abspath(rel))

    def listdir(self, rel: str) -> list[str]:
        """Sorted entry names of a directory; empty when missing."""
        try:
            return sorted(os.listdir(self.abspath(rel)))
        except OSError:
            return []

    def readlink(self, rel: str) -> str | None:
        try:
            return os.readlink(self.abspath(rel))
        except OSError:
            return None

    def realpath(self, rel: str) -> str | None:
        """Canonical path of ``rel`` relative to the root (symlinks resolved)."""
        p = os.path.realpath(self.abspath(rel))
        base = os.path.realpath(self.root)
        if p != base and not p.startswith(base.rstrip("/") + "/"):
            return None
        return os.path.relpath(p, base)


class RecordingSysfs(Sysfs):
    """A ``Sysfs`` that mirrors everything it reads into ``out`` (fixture capture).

    Files are written at their canonical location; symlinked path components
    (``bus/pci/devices/<bdf>``, ``driver``) are recreated as relative symlinks, so
    ``Sysfs(out)`` answers the same calls with the same results.
    """

    def __init__(
        self, root: str | os.PathLike[str], out: str | os.PathLike[str]
    ) -> None:
        super().__init__(root)
        self.out = os.fspath(out).rstrip("/")

    def _mirror_links(self, rel: str) -> None:
        parts = rel.strip("/").split("/")
        cur = ""
        for part in parts:
            cur = f"{cur}/{part}" if cur else part
            src = self.abspath(cur)
            if os.path.islink(src):
                dst = f"{self.out}/{cur}"
                if not os.path.lexists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.symlink(os.readlink(src), dst)

    def read_bytes(self, rel: str, limit: int = _MAX_READ) -> bytes | None:
        raw = super().read_bytes(rel, limit)
        if raw is None:
            return None
        self._mirror_links(rel)
        real = self.realpath(rel)
        if real is not None:
            dst = f"{self.out}/{real}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(raw)
        return raw

    def listdir(self, rel: str) -> list[str]:
        names = super().listdir(rel)
        if not names:
            return names
        self._mirror_links(rel)
        real = self.realpath(rel)
        if real is not None:
            os.makedirs(f"{self.out}/{real}", exist_ok=True)
        for name in names:
            child = f"{rel.rstrip('/')}/{name}"
            if os.path.islink(self.abspath(child)):
                self._mirror_links(child)
            elif real is not None and os.path.isdir(self.abspath(child)):
                os.makedirs(f"{self.out}/{real}/{name}", exist_ok=True)
        return names

    def readlink(self, rel: str) -> str | None:
        target = super().readlink(rel)
        if target is not None:
            self._mirror_links(rel)
        return target

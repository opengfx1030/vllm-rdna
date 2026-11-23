# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GHCR cleanup keeps what a retained tag references and deletes the rest.

The script runs against a fake registry: versions with digests, tags and
creation times, plus a manifest graph. No network is used.
"""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".github/workflows/scripts/prune_package_versions.py"
)
NOW = datetime.now(UTC)
OLD = NOW - timedelta(days=30)
YOUNG = NOW - timedelta(days=1)


@pytest.fixture
def module(monkeypatch):
    spec = importlib.util.spec_from_file_location("prune_package_versions", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules[__module__].
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def _version(mod, vid, tags, created):
    return mod.Version(
        id=vid, digest=f"sha256:{vid}", tags=tuple(tags), created_at=created
    )


def _fake_registry(mod, monkeypatch, versions, children):
    """Children maps a digest to the digests of its child manifests."""
    deleted: list[int] = []

    def manifest(self, reference):
        digest = reference
        if not reference.startswith("sha256:"):
            digest = next(v.digest for v in versions if reference in v.tags)
        kids = children.get(digest, [])
        data: dict[str, Any] = {"manifests": [{"digest": c} for c in kids]}
        if not kids:
            data = {"config": {"size": 1}, "layers": []}
        return mod.Manifest(digest=digest, data=data)

    monkeypatch.setattr(mod, "load_versions", lambda owner, package: versions)
    monkeypatch.setattr(mod, "registry_token", lambda image: "token")
    monkeypatch.setattr(mod.Registry, "manifest", manifest)
    monkeypatch.setattr(
        mod,
        "api_delete",
        lambda path: deleted.append(int(path.rsplit("/", 1)[1])),
    )
    return deleted


def _graph(mod, monkeypatch):
    """A superseded sha-tagged build, the live nightly, a young build, a release.

    ids: 1 old sha-only index (children 2, 3), 4 live index tagged
    sha-live + nightly (children 5, 6), 7 old untagged superseded index
    (children 8, 9), 10 young sha-only index, 11 release v1 (children 12, 13).
    """
    versions = [
        _version(mod, 1, ["sha-old"], OLD),
        _version(mod, 2, [], OLD),
        _version(mod, 3, [], OLD),
        _version(mod, 4, ["sha-live", "nightly"], OLD),
        _version(mod, 5, [], OLD),
        _version(mod, 6, [], OLD),
        _version(mod, 7, [], OLD),
        _version(mod, 8, [], OLD),
        _version(mod, 9, [], OLD),
        _version(mod, 10, ["sha-young"], YOUNG),
        _version(mod, 11, ["v1"], OLD),
        _version(mod, 12, [], OLD),
        _version(mod, 13, [], OLD),
    ]
    children = {
        "sha256:1": ["sha256:2", "sha256:3"],
        "sha256:4": ["sha256:5", "sha256:6"],
        "sha256:7": ["sha256:8", "sha256:9"],
        "sha256:10": [],
        "sha256:11": ["sha256:12", "sha256:13"],
    }
    return versions, _fake_registry(mod, monkeypatch, versions, children)


def test_default_patterns_delete_old_sha_builds_and_superseded_digests(
    module, monkeypatch
):
    _, deleted = _graph(module, monkeypatch)
    ok = module.prune("o", "p", 7, dry_run=False, disposable_tags=("sha-*",))
    assert ok
    # old sha-only index and its children, and the superseded untagged index
    # with its children; nothing that nightly, a release or a young build holds.
    assert sorted(deleted) == [1, 2, 3, 7, 8, 9]


def test_without_patterns_only_unreferenced_untagged_versions_go(module, monkeypatch):
    _, deleted = _graph(module, monkeypatch)
    ok = module.prune("o", "p", 7, dry_run=False, disposable_tags=())
    assert ok
    assert sorted(deleted) == [7, 8, 9]


def test_dry_run_deletes_nothing(module, monkeypatch):
    _, deleted = _graph(module, monkeypatch)
    assert module.prune("o", "p", 7, dry_run=True, disposable_tags=("sha-*",))
    assert deleted == []


def test_a_version_with_another_tag_is_never_disposable(module):
    shared = _version(module, 1, ["sha-live", "nightly"], OLD)
    assert not shared.disposable(("sha-*",))
    assert _version(module, 2, ["sha-a", "sha-b"], OLD).disposable(("sha-*",))
    assert _version(module, 3, [], OLD).disposable(())
    assert not _version(module, 4, ["v1"], OLD).disposable(("sha-*",))

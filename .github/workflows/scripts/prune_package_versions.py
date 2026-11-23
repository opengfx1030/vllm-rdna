#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prune untagged GHCR package versions that no retained image references.

GHCR records every manifest pushed to a package as its own package version. A
tag does not own the manifests it points at: the tag names an OCI image index
whose platform manifests and buildx provenance attestation are separate,
untagged versions of the same package. Deleting those versions breaks
``docker pull`` of the tagged image, which is why the plain "delete untagged
versions" APIs cannot be used for these packages.

This script deletes an untagged version only when it is older than
``RETENTION_DAYS`` *and* no retained version references it: not shared with a
tagged version, not a child of a tagged index (platform image, provenance or
SBOM attestation) and not a child of an untagged index that is itself retained.

Tagged versions are never deleted. The decision is made from the registry's
manifest graph, so it stays correct if the image layout changes (more
platforms, SBOM attestations, cosign signatures stored as index children).

Environment:
    PACKAGES: space- or comma-separated package names (default ``vllm-rdna``)
    PACKAGE_OWNER: package owner (default ``opengfx1030``)
    RETENTION_DAYS: minimum age of a deletable version (default ``7``)
    GITHUB_EVENT_NAME: a manual run defaults to a dry run, other events delete
    INPUT_DRY_RUN: ``true``/``false`` forces the mode, unset keeps the default
    GH_TOKEN / GITHUB_TOKEN: token with ``packages: read`` (``write`` to delete)

Manifests are read from the registry with an anonymous pull token.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
USER_AGENT = "vllm-rdna-package-cleanup"
REGISTRY_HOST = os.environ.get("REGISTRY_HOST", "ghcr.io")

INDEX_MEDIA_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
MANIFEST_ACCEPT = ",".join(
    sorted(
        INDEX_MEDIA_TYPES
        | {
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        }
    )
)


class PackageCleanupError(RuntimeError):
    """An unexpected answer that must stop the run instead of guessing."""


def api_token() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise PackageCleanupError("GH_TOKEN or GITHUB_TOKEN is required")
    return token


def http_get(
    url: str,
    *,
    token: str | None = None,
    method: str = "GET",
    accept: str | None = None,
) -> tuple[dict[str, str], bytes]:
    headers = {"User-Agent": USER_AGENT, "X-GitHub-Api-Version": API_VERSION}
    if accept:
        headers["Accept"] = accept
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return (
                {key.lower(): value for key, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as error:
        raise PackageCleanupError(
            f"{method} {url} failed: {error.code} {error.reason}"
        ) from error


def next_link(link_header: str) -> str:
    for part in link_header.split(","):
        sections = part.split(";")
        if any(section.strip() == 'rel="next"' for section in sections[1:]):
            return sections[0].strip().removeprefix("<").removesuffix(">")
    return ""


def api_get(path: str) -> list[dict[str, Any]]:
    """Return every page of a GitHub REST list endpoint."""
    url = f"{API_ROOT}{path}"
    token = api_token()
    items: list[dict[str, Any]] = []
    while url:
        headers, body = http_get(url, token=token, accept="application/vnd.github+json")
        items.extend(json.loads(body))
        url = next_link(headers.get("link", ""))
    return items


def api_delete(path: str) -> None:
    http_get(f"{API_ROOT}{path}", token=api_token(), method="DELETE")


def registry_token(image: str) -> str:
    """Fetch an anonymous pull token for ``image``.

    GHCR serves these for public packages, which is what the image is. If the
    package ever becomes private the token request fails and the run stops
    before anything is deleted.
    """
    scope = urllib.parse.quote(f"repository:{image}:pull", safe=":/")
    url = f"https://{REGISTRY_HOST}/token?service={REGISTRY_HOST}&scope={scope}"
    _, body = http_get(url)
    return str(json.loads(body)["token"])


@dataclass(frozen=True)
class Manifest:
    digest: str
    data: dict[str, Any]

    @property
    def children(self) -> list[dict[str, Any]]:
        return list(self.data.get("manifests") or [])

    @property
    def is_index(self) -> bool:
        return (
            "manifests" in self.data or self.data.get("mediaType") in INDEX_MEDIA_TYPES
        )

    @property
    def own_bytes(self) -> int:
        """Config plus layer bytes named by this manifest (index bytes excluded)."""
        config = self.data.get("config") or {}
        return int(config.get("size", 0)) + sum(
            int(layer.get("size", 0)) for layer in self.data.get("layers") or []
        )


@dataclass
class Registry:
    image: str
    token: str
    manifests: dict[str, Manifest] = field(default_factory=dict)

    def manifest(self, reference: str) -> Manifest:
        if reference not in self.manifests:
            headers, body = http_get(
                f"https://{REGISTRY_HOST}/v2/{self.image}/manifests/{reference}",
                token=self.token,
                accept=MANIFEST_ACCEPT,
            )
            self.manifests[reference] = Manifest(
                digest=headers.get("docker-content-digest", reference),
                data=json.loads(body),
            )
        return self.manifests[reference]


@dataclass(frozen=True)
class Version:
    id: int
    digest: str
    tags: tuple[str, ...]
    created_at: datetime

    @property
    def tagged(self) -> bool:
        return bool(self.tags)

    @property
    def label(self) -> str:
        tags = ", ".join(self.tags)
        return tags or f"untagged version {self.id}"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_versions(owner: str, package: str) -> list[Version]:
    """Return the package versions with their digests, tags and creation times."""
    versions: list[Version] = []
    for item in api_get(
        f"/orgs/{owner}/packages/container/{package}/versions?per_page=100"
    ):
        digest = str(item["name"])
        if not digest.startswith("sha256:"):
            raise PackageCleanupError(
                f"version {item['id']} is not identified by a digest: {digest!r}"
            )
        container = (item.get("metadata") or {}).get("container") or {}
        versions.append(
            Version(
                id=int(item["id"]),
                digest=digest,
                tags=tuple(container.get("tags") or ()),
                created_at=parse_time(str(item["created_at"])),
            )
        )
    return versions


def prune(owner: str, package: str, retention_days: int, dry_run: bool) -> bool:
    """Prune one package; return False when a version could not be deleted."""
    image = f"{owner}/{package}"
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    versions = load_versions(owner, package)
    registry = Registry(image=image, token=registry_token(image))

    # The registry is the source of truth for tag -> digest; the package API is
    # a cached view. Refuse to delete anything if the two disagree.
    for version in versions:
        for tag in version.tags:
            resolved = registry.manifest(tag).digest
            if resolved != version.digest:
                raise PackageCleanupError(
                    f"{image}:{tag} resolves to {resolved}, but version "
                    f"{version.id} reports {version.digest}"
                )

    untagged = [version for version in versions if not version.tagged]
    expired = [version for version in untagged if version.created_at < cutoff]

    # Reachability closure: a version survives if it is tagged or too young, and
    # everything such a version references survives with it. Only versions
    # outside that closure can be deleted.
    retained: dict[str, str] = {
        version.digest: version.label
        for version in versions
        if version.tagged or version.created_at >= cutoff
    }
    referenced: dict[str, set[str]] = {}
    pending = list(retained)
    while pending:
        parent_digest = pending.pop()
        parent = retained[parent_digest]
        for child in registry.manifest(parent_digest).children:
            child_digest = str(child["digest"])
            annotations = child.get("annotations") or {}
            kind = annotations.get("vnd.docker.reference.type")
            referrer = f"{parent} ({kind})" if kind else parent
            referenced.setdefault(child_digest, set()).add(referrer)
            if child_digest not in retained:
                retained[child_digest] = referrer
                pending.append(child_digest)

    protected = [version for version in expired if version.digest in retained]
    candidates = sorted(
        (version for version in expired if version.digest not in retained),
        key=lambda version: (
            not registry.manifest(version.digest).is_index,
            version.created_at,
            version.id,
        ),
    )
    freed = sum(registry.manifest(version.digest).own_bytes for version in candidates)

    print(f"package: {image} on {REGISTRY_HOST}")
    print(f"retention: {retention_days} days, cutoff {cutoff:%Y-%m-%dT%H:%M:%SZ}")
    print(
        f"versions: {len(versions)} total, {len(versions) - len(untagged)} tagged, "
        f"{len(untagged)} untagged, {len(expired)} untagged and expired"
    )
    print(f"referenced digests held by retained versions: {len(referenced)}")

    if dry_run:
        for version in sorted(protected, key=lambda item: item.created_at):
            referrers = ", ".join(sorted(referenced.get(version.digest, {"unknown"})))
            print(
                f"  keep {version.id} {version.digest} "
                f"{version.created_at:%Y-%m-%dT%H:%M:%SZ} referenced by {referrers}"
            )
    else:
        print(
            f"held back {len(protected)} expired untagged versions that a "
            "retained version still references"
        )

    verb = "would delete" if dry_run else "delete"
    for version in candidates:
        manifest = registry.manifest(version.digest)
        kind = "index" if manifest.is_index else "image"
        print(
            f"  {verb} {version.id} {version.digest} "
            f"{version.created_at:%Y-%m-%dT%H:%M:%SZ} {kind} "
            f"{manifest.own_bytes} bytes"
        )

    note = "(layer bytes; blobs shared with retained images stay)"
    if dry_run:
        print(f"would delete {len(candidates)} versions, {freed} bytes {note}")
        return True

    failures = 0
    for version in candidates:
        path = f"/orgs/{owner}/packages/container/{package}/versions/{version.id}"
        try:
            api_delete(path)
        except PackageCleanupError as error:
            failures += 1
            print(f"::error::could not delete version {version.id}: {error}")
    print(
        f"deleted {len(candidates) - failures} of {len(candidates)} versions, "
        f"{freed} bytes {note}"
    )
    return not failures


def main() -> int:
    event = os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch")
    requested = (os.environ.get("INPUT_DRY_RUN") or "").strip().lower()
    if requested in {"true", "false"}:
        # An explicit input always wins: it is what a manual run asked for and
        # what the read-only probe sets.
        dry_run = requested == "true"
    else:
        # Scheduled runs delete; manual runs default to a dry run.
        dry_run = event == "workflow_dispatch"
    try:
        retention_days = int(os.environ.get("RETENTION_DAYS") or 7)
    except ValueError:
        print("::error::RETENTION_DAYS must be an integer")
        return 2
    if retention_days < 1:
        print("::error::RETENTION_DAYS must be at least 1")
        return 2

    owner = os.environ.get("PACKAGE_OWNER", "opengfx1030")
    packages = [
        name
        for name in os.environ.get("PACKAGES", "vllm-rdna").replace(",", " ").split()
        if name
    ]
    print(f"mode: {'dry run, nothing is deleted' if dry_run else 'delete'}")

    ok = True
    for package in packages:
        try:
            ok = prune(owner, package, retention_days, dry_run) and ok
        except PackageCleanupError as error:
            ok = False
            print(f"::error::{owner}/{package}: {error}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

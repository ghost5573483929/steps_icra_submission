"""Small, deterministic provenance helpers for derived publication artifacts."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def sha256_file(path: str | Path) -> str:
    """Return ``"sha256:" + hexdigest`` of a single file's raw bytes, read in fixed-size chunks."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def implementation_hash(paths: Iterable[str | Path], *, base: str | Path) -> str:
    """Hash the concatenated contents of exactly the given files, keyed by base-relative path.

    Content-addressed over exactly the listed paths: imports are not followed, so closure
    over everything that affects the output is the caller's responsibility.

    Files are iterated by resolved absolute path for deterministic ordering, but the label
    folded into the digest is each file's path relative to ``base``, so the same files reached
    by different routes hash identically. A path outside ``base`` is labelled by bare filename.
    """
    base = Path(base).resolve()
    hasher = hashlib.sha256()
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        try:
            label = path.relative_to(base).as_posix()
        except ValueError:
            label = path.name
        hasher.update(label.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def git_provenance(root: str | Path, timeout: float = 5.0) -> dict[str, Any]:
    """Capture this repository's commit and uncommitted-change status for run provenance.

    Returns ``{"git_commit": <sha>, "git_dirty": <bool>}``. Any resolution failure is
    warned about and reported as ``{"git_commit": "unknown", "git_dirty": None}`` rather
    than raised, so a missing provenance field never aborts a conversion run.
    """
    root = Path(root)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                 text=True, timeout=timeout, check=True).stdout.strip()
        porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                    text=True, timeout=timeout, check=True).stdout
        return {"git_commit": commit, "git_dirty": bool(porcelain.strip())}
    except Exception as error:
        logger.warning("Could not resolve git provenance for %s: %r", root, error)
        return {"git_commit": "unknown", "git_dirty": None}


def source_set_revision(paths: Iterable[str | Path], *, base: str | Path) -> str:
    """Hash exact file identities and revisions without rereading large Parquet bodies.

    Identity-addressed, not content-addressed: each row is ``(relative path, size_bytes,
    mtime_ns)`` and the file bytes are never read. A value is therefore **not stable across
    clones or machines** and changes on an mtime-only touch, which publication audits treat
    as stale metadata. Switching this to a content-addressed basis would change every
    published ``canonical_source_set_revision``, so it needs a human decision.
    """
    base = Path(base).resolve()
    rows = []
    for value in sorted((Path(item).resolve() for item in paths), key=str):
        stat = value.stat()
        try:
            label = value.relative_to(base).as_posix()
        except ValueError:
            label = value.name
        rows.append(
            {
                "path": label,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()

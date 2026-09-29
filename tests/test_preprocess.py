from __future__ import annotations

import subprocess
from pathlib import Path

from artifact_provenance import git_provenance
from preprocess import git_provenance as preprocess_git_provenance


def test_git_provenance_reports_commit_and_dirty_status_in_this_repo() -> None:
    assert preprocess_git_provenance is git_provenance

    root = Path(__file__).resolve().parents[1]
    provenance = git_provenance(root)

    assert set(provenance) == {"git_commit", "git_dirty"}
    assert isinstance(provenance["git_commit"], str)
    assert provenance["git_commit"] != "unknown"
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert provenance["git_commit"] == expected_commit
    assert isinstance(provenance["git_dirty"], bool)


def test_git_provenance_degrades_gracefully_outside_a_git_checkout(tmp_path: Path) -> None:
    provenance = git_provenance(tmp_path, timeout=5.0)

    assert provenance == {"git_commit": "unknown", "git_dirty": None}


def test_git_provenance_degrades_gracefully_on_timeout(tmp_path: Path) -> None:
    provenance = git_provenance(tmp_path, timeout=1e-9)

    assert provenance == {"git_commit": "unknown", "git_dirty": None}

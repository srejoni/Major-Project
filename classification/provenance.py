"""
classification/provenance.py

Single source of truth for two things that were previously either
hardcoded or silently identical across runs:

1. Which *exact* trained checkpoint backs a given architecture in the
   MODEL_REGISTRY (sha256 + training timestamp + training git commit +
   which dataset manifest it was trained against), so two exports made
   before/after a checkpoint swap are provably different.

2. What pipeline code produced this export (PIPELINE_VERSION), derived
   from the current git commit of the inference code rather than a
   constant that never changes.

Both scripts/push_checkpoint.py (writer) and assemble_study_output.py
(reader, via _model_versions_block()) import this module so there is
exactly one manifest format and one version-computation rule. Place
this file at classification/provenance.py — it assumes it lives one
directory below the repo root (adjust REPO_ROOT below if that changes).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parents[1]                 # .../PDSCD
WEIGHTS_DIR = REPO_ROOT / "classification" / "weights"
ARCHIVE_DIR = WEIGHTS_DIR / "archive"
MANIFEST_PATH = WEIGHTS_DIR / "checkpoint_manifest.json"

# Bump this manually only for a genuinely breaking pipeline change you
# want reflected even if git history is ever unavailable (e.g. a
# detached tarball export). Day-to-day code changes are already
# captured automatically via the git-commit suffix in
# get_pipeline_version() below — you should rarely need to touch this.
PIPELINE_VERSION_BASE = "1.0.0"


# ---------------------------------------------------------------------------
# Checkpoint hashing
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through sha256 so multi-hundred-MB checkpoints
    don't get loaded into memory whole."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Git helpers — degrade gracefully rather than crash the export
# ---------------------------------------------------------------------------
def get_git_commit(short: bool = True, cwd: Optional[Path] = None) -> str:
    """Current HEAD commit of the repo at `cwd` (defaults to REPO_ROOT).
    Returns 'unknown-nogit' instead of raising if git isn't available or
    `cwd` isn't inside a git repo (e.g. a frozen Kaggle dataset
    snapshot) — provenance should degrade, not break inference."""
    args = ["git", "rev-parse"]
    args.append("--short" if short else "HEAD")
    if short:
        args.append("HEAD")
    try:
        out = subprocess.run(
            args,
            cwd=str(cwd or REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        commit = out.stdout.strip()
        return commit if commit else "unknown-nogit"
    except Exception:
        return "unknown-nogit"


def is_git_dirty(cwd: Optional[Path] = None) -> bool:
    """True if there are uncommitted changes. Used to warn/refuse when
    pushing a checkpoint against a dirty tree, since the resulting
    training_git_commit provenance would otherwise be misleading."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd or REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Checkpoint manifest
# ---------------------------------------------------------------------------
@dataclass
class CheckpointRecord:
    model_name: str               # e.g. "efficientnet_b3" — must match MODEL_REGISTRY key
    archived_filename: str        # immutable, e.g. "efficientnet_b3_a1b2c3d4_20260918T101500Z.pt"
    committed_filename: str       # stable name inference actually loads, e.g. "efficientnet_b3_best.pt"
    sha256: str
    source_checkpoint_path: str   # original ephemeral CHECKPOINT_DIR path at push time
    epoch: Optional[int]
    val_loss: Optional[float]
    dataset_manifest_path: str    # train.py's MANIFEST_PATH at push time
    dataset_manifest_rows: Optional[int]
    training_git_commit: str      # repo commit at push time (best available proxy for
                                   # "what code produced this checkpoint" — train.py doesn't
                                   # currently self-stamp its own commit into the .pt file)
    pushed_at: str                 # ISO8601 UTC


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"models": {}}  # model_name -> list[CheckpointRecord dict], newest last
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)  # atomic on POSIX — never leaves a half-written manifest


def record_checkpoint(record: CheckpointRecord) -> None:
    """Append `record` under its model_name (history kept, newest last)
    and atomically rewrite the manifest. Does NOT git-commit — that is
    scripts/push_checkpoint.py's job, after this succeeds."""
    manifest = load_manifest()
    manifest.setdefault("models", {}).setdefault(record.model_name, [])
    manifest["models"][record.model_name].append(asdict(record))
    _atomic_write_json(MANIFEST_PATH, manifest)


def get_latest_checkpoint_record(model_name: str) -> Optional[dict]:
    manifest = load_manifest()
    entries = manifest.get("models", {}).get(model_name, [])
    return entries[-1] if entries else None


# ---------------------------------------------------------------------------
# Pipeline version
# ---------------------------------------------------------------------------
def get_pipeline_version() -> str:
    """Replaces the old hardcoded PIPELINE_VERSION constant.
    Format: '<manually-bumped base>+<git short sha>[.dirty]'
    e.g. '1.0.0+a1b2c3d' or '1.0.0+a1b2c3d.dirty'.
    Two exports made from different code states are now always
    distinguishable; a dirty tree is flagged rather than hidden."""
    commit = get_git_commit(short=True)
    suffix = ".dirty" if is_git_dirty() else ""
    return f"{PIPELINE_VERSION_BASE}+{commit}{suffix}"

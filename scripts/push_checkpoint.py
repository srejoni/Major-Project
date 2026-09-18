#!/usr/bin/env python3
"""
scripts/push_checkpoint.py

Closes T2 item 2: copies the best checkpoint train.py just produced out
of the ephemeral CHECKPOINT_DIR into the committed classification/weights/
directory, records real provenance for it (checkpoint sha256, epoch,
val_loss, which manifest it trained against, training-time git commit),
and pushes to git. Skip running this after a retrain and the old
checkpoint keeps silently serving.

Usage:
    python scripts/push_checkpoint.py \
        --model-name efficientnet_b3 \
        --checkpoint /kaggle/working/checkpoints/best_efficientnet_b3.pt \
        --epoch 5 --val-loss 0.5635 \
        --manifest-path /kaggle/input/datasets/elenoremadams/axial-laterality-fix-crops/manifest.csv

Run this manually right after a training run you intend to ship. It is
deliberately NOT auto-invoked at the end of train.py — an experimental
run you're not shipping shouldn't silently overwrite the served
checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classification"))
from provenance import (  # noqa: E402
    REPO_ROOT, WEIGHTS_DIR, ARCHIVE_DIR,
    CheckpointRecord, sha256_file, get_git_commit, is_git_dirty,
    record_checkpoint,
)


def count_manifest_rows(manifest_path: Path) -> "int | None":
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # minus header row


def run(cmd: list, cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True,
                     help="e.g. efficientnet_b3 / convnext_tiny / resnet34 "
                          "— must match the MODEL_REGISTRY key")
    ap.add_argument("--checkpoint", required=True, type=Path,
                     help="path to the ephemeral .pt in train.py's CHECKPOINT_DIR")
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--val-loss", type=float, default=None)
    ap.add_argument("--manifest-path", type=Path, default=None,
                     help="the training manifest.csv this checkpoint was trained against")
    ap.add_argument("--no-push", action="store_true",
                     help="commit locally but skip `git push` (review first)")
    ap.add_argument("--force", action="store_true",
                     help="proceed even if the working tree has other uncommitted "
                          "changes (their content becomes part of this commit's "
                          "provenance too, so use deliberately)")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    if is_git_dirty() and not args.force:
        print(
            "ERROR: working tree has uncommitted changes unrelated to this "
            "checkpoint push. Commit/stash them first, or re-run with --force "
            "if you accept them being folded into this commit's provenance.",
            file=sys.stderr,
        )
        return 1

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Hashing {args.checkpoint} ...")
    digest = sha256_file(args.checkpoint)
    short_hash = digest[:8]
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")

    archived_filename = f"{args.model_name}_{short_hash}_{timestamp}.pt"
    committed_filename = f"{args.model_name}_best.pt"  # stable name inference loads

    archived_path = ARCHIVE_DIR / archived_filename
    committed_path = WEIGHTS_DIR / committed_filename

    print(f"Writing immutable archive copy: {archived_path}")
    archived_path.write_bytes(args.checkpoint.read_bytes())

    print(f"Writing stable committed copy: {committed_path}")
    committed_path.write_bytes(args.checkpoint.read_bytes())

    manifest_rows = count_manifest_rows(args.manifest_path) if args.manifest_path else None
    training_commit = get_git_commit(short=True)

    record = CheckpointRecord(
        model_name=args.model_name,
        archived_filename=archived_filename,
        committed_filename=committed_filename,
        sha256=digest,
        source_checkpoint_path=str(args.checkpoint),
        epoch=args.epoch,
        val_loss=args.val_loss,
        dataset_manifest_path=str(args.manifest_path) if args.manifest_path else "unknown",
        dataset_manifest_rows=manifest_rows,
        training_git_commit=training_commit,
        pushed_at=now.isoformat(),
    )
    record_checkpoint(record)
    print(f"Recorded provenance in {WEIGHTS_DIR / 'checkpoint_manifest.json'}")

    rel_paths = [
        str(archived_path.relative_to(REPO_ROOT)),
        str(committed_path.relative_to(REPO_ROOT)),
        str((WEIGHTS_DIR / "checkpoint_manifest.json").relative_to(REPO_ROOT)),
    ]
    run(["git", "add"] + rel_paths, cwd=REPO_ROOT)
    commit_msg = (
        f"checkpoint: {args.model_name} {short_hash} "
        f"(epoch={args.epoch}, val_loss={args.val_loss})"
    )
    run(["git", "commit", "-m", commit_msg], cwd=REPO_ROOT)

    if args.no_push:
        print("Committed locally. Skipping push (--no-push). Review, then "
              "`git push` manually.")
    else:
        run(["git", "push"], cwd=REPO_ROOT)
        print("Pushed.")

    print(
        "\nDone. classification/weights/" + committed_filename +
        " now points at sha256 " + short_hash + ".\n"
        "Confirm assemble_study_output.py's _model_versions_block() reflects "
        "this by re-running a study export and checking the emitted sha256 "
        "changed from the previous export."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

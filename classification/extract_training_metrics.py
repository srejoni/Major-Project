"""
classification/extract_training_metrics.py

Recovers the best-checkpoint epoch/val_loss/val_acc from a pasted train.py
console log, matched exactly against train.py's real print statements --
not guessed:

  per-epoch line (main(), inside the epoch loop):
    "[epoch {epoch}/{NUM_EPOCHS}] train_loss={train_loss:.4f}  "
    "val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.2e}  "
    "({elapsed:.1f}s)"

  best-checkpoint-saved line (main(), only printed when val_loss improves):
    "[train] new best val_loss={val_loss:.4f} -- saved {ckpt_path}"

  final line (main(), after the loop ends -- early stopping or NUM_EPOCHS
  reached):
    "[train] training complete. model_name={args.model_name}  "
    "best_val_loss={best_val_loss:.4f}"

IMPORTANT, non-obvious from the code: the "new best" line does NOT carry
val_acc or the epoch number, and the "training complete" line does not
carry val_acc either. val_acc only ever appears on the per-epoch line. So
recovering "epoch X, val_loss Y, val_acc Z" for the checkpoint that
actually got saved to disk means: find the LAST "[train] new best
val_loss=" line (that's the save that's still on disk -- an early-stopped
run's checkpoint is whichever save happened last, not the final epoch run),
then read val_acc and the epoch number off the per-epoch line immediately
associated with it (same val_loss value, by construction -- both lines
print in the same loop iteration, epoch line first).

Usage:
    python classification/extract_training_metrics.py path/to/pasted_log.txt
    # or pipe it in:
    cat pasted_log.txt | python classification/extract_training_metrics.py
"""
import re
import sys

EPOCH_RE = re.compile(
    r"\[epoch (\d+)/(\d+)\]\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+"
    r"val_acc=([\d.]+)\s+lr=([\d.eE+\-]+)\s+\(([\d.]+)s\)"
)
BEST_RE = re.compile(r"\[train\] new best val_loss=([\d.]+) -- saved (.+)")
COMPLETE_RE = re.compile(
    r"\[train\] training complete\.\s+model_name=(\S+)\s+best_val_loss=([\d.]+)"
)


def extract_best_checkpoint_metrics(log_text: str) -> dict:
    epochs = []
    best_saves = []
    completion = None

    for line in log_text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epochs.append({
                "epoch": int(m.group(1)),
                "num_epochs": int(m.group(2)),
                "train_loss": float(m.group(3)),
                "val_loss": float(m.group(4)),
                "val_acc": float(m.group(5)),
                "lr": m.group(6),
                "elapsed_s": float(m.group(7)),
            })
            continue

        m = BEST_RE.search(line)
        if m:
            best_saves.append({
                "val_loss": float(m.group(1)),
                "checkpoint_path": m.group(2).strip(),
            })
            continue

        m = COMPLETE_RE.search(line)
        if m:
            completion = {
                "model_name": m.group(1),
                "best_val_loss": float(m.group(2)),
            }

    if not best_saves:
        raise ValueError(
            "No '[train] new best val_loss=...' line found in this log. Either "
            "this isn't a train.py console log, or the log was truncated before "
            "the first improving epoch printed (unlikely -- epoch 1 is always "
            "an improvement over float('inf'))."
        )

    # The LAST "new best" save is what's actually sitting on disk -- early
    # stopping means later, non-improving epochs ran but didn't overwrite it.
    last_best = best_saves[-1]

    matching_epochs = [e for e in epochs if abs(e["val_loss"] - last_best["val_loss"]) < 1e-4]
    if not matching_epochs:
        raise ValueError(
            f"Found '[train] new best val_loss={last_best['val_loss']}' but no "
            f"per-epoch line with a matching val_loss -- log is likely truncated "
            f"or edited. Paste the full, unedited log."
        )
    best_epoch = matching_epochs[-1]

    if completion is not None and abs(completion["best_val_loss"] - best_epoch["val_loss"]) > 1e-4:
        print(
            f"WARNING: '[train] training complete' reported best_val_loss="
            f"{completion['best_val_loss']}, which doesn't match the last saved "
            f"checkpoint's val_loss ({best_epoch['val_loss']}). This log may be "
            f"incomplete, or from a different/interrupted run than the checkpoint "
            f"you're about to push -- double check before using these numbers.",
            file=sys.stderr,
        )

    return {
        "epoch": best_epoch["epoch"],
        "val_loss": best_epoch["val_loss"],
        "val_acc": best_epoch["val_acc"],
        "checkpoint_path": last_best["checkpoint_path"],
        "model_name": completion["model_name"] if completion else None,
    }


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    r = extract_best_checkpoint_metrics(text)

    print(f"model_name   = {r['model_name']}")
    print(f"epoch        = {r['epoch']}")
    print(f"val_loss     = {r['val_loss']:.4f}")
    print(f"val_acc      = {r['val_acc']:.4f}")
    print(f"checkpoint   = {r['checkpoint_path']}")
    print()
    print("For your notes (same convention as EffNet-B3's 'ep4, val_loss 0.5481, "
          f"val_acc 0.8046'):")
    print(f"  ep{r['epoch']}, val_loss {r['val_loss']:.4f}, val_acc {r['val_acc']:.4f}")
    print()
    print("For push_checkpoint.py:")
    print(f"  --epoch {r['epoch']} --val-loss {r['val_loss']:.4f}")


if __name__ == "__main__":
    main()

"""
classification/verify_retrain_delta.py

Read-only comparison of a retrained checkpoint against its pre-retrain
baseline, on the SAME validation rows, to produce the two numbers Phase C
actually asks for and that verify_training_quality.py doesn't print:

  - pooled Moderate recall      (that script only prints per-condition)
  - lateralized-accuracy-gap    (canal accuracy minus the mean accuracy of
                                  the four lateralized conditions -- a
                                  different quantity from
                                  verify_training_quality.py's axial-vs-
                                  sagittal macro-F1 gap)

Also reports weighted NLL for both checkpoints on identical rows, since the
retrain's logged val_loss (0.8688, ep6) reads worse than the pre-retrain
run's (0.5481, ep4) and nothing in the docs says whether that's a real
regression or a comparison artifact from the crops that changed under the
laterality fix.

Does NOT retrain anything, does NOT modify verify_training_quality.py (its
confusion-matrix / recall / macro-F1 functions are imported directly so the
two scripts can't drift on definitions), and does NOT open the deferred
32/10 left/right skew investigation -- a NOT_IMPROVED verdict here just
gets recorded; escalating it is a separate, later decision.
"""
import argparse
import hashlib
import json
import sys

import torch
import numpy as np
from torchvision.models import efficientnet_b3

sys.path.append("/kaggle/working/PDSCD")
from classification.build_dataloaders import build_classification_dataloaders
from classification.verify_training_quality import (
    confusion_matrix_3x3,
    per_class_recall,
    macro_f1,
    collect_predictions,
)

CANAL_CONDITION = "spinal_canal_stenosis"
FORAMINAL_CONDITIONS = {"left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"}
SUBARTICULAR_CONDITIONS = {"left_subarticular_stenosis", "right_subarticular_stenosis"}
LATERALIZED_CONDITIONS = FORAMINAL_CONDITIONS | SUBARTICULAR_CONDITIONS
ALL_CONDITIONS = {CANAL_CONDITION} | LATERALIZED_CONDITIONS

SEVERITY_LABELS = ["Normal/Mild", "Moderate", "Severe"]
MODERATE_CLASS_IDX = 1
CLASS_WEIGHTS = [1.0, 2.0, 4.0]  # matches train.py's CrossEntropyLoss(weight=...)

EXPECTED_VAL_ROW_RANGE = (10311, 10353)     # 10,353 documented pre-migration, minus <=42 lost rows
DOCUMENTED_CANAL_ACC_RANGE = (0.84, 0.91)   # 0.86-0.89 +/- 0.02 backstop band; canal crops untouched by migration


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint_path, device):
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, 3)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def condition_accuracy(cm):
    return float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else None


def per_condition_metrics(by_condition):
    """condition -> {n, accuracy, macro_f1, recalls[3], cm}"""
    out = {}
    for cond, (y_true, y_pred) in by_condition.items():
        cm = confusion_matrix_3x3(y_true, y_pred)
        out[cond] = {
            "n": int(cm.sum()),
            "accuracy": condition_accuracy(cm),
            "macro_f1": macro_f1(cm),
            "recalls": per_class_recall(cm),
            "cm": cm,
        }
    return out


def pooled_moderate_recall(by_condition):
    """Pool y_true/y_pred across every condition, then recall for class 1 (Moderate)."""
    y_true_all, y_pred_all = [], []
    for y_true, y_pred in by_condition.values():
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
    cm = confusion_matrix_3x3(y_true_all, y_pred_all)
    recalls = per_class_recall(cm)
    pooled_acc = condition_accuracy(cm)
    return recalls[MODERATE_CLASS_IDX], int(cm[MODERATE_CLASS_IDX].sum()), pooled_acc


def lateralized_gap(metrics, group):
    """Canal accuracy minus mean accuracy of `group` conditions. None if either side is missing."""
    if CANAL_CONDITION not in metrics or metrics[CANAL_CONDITION]["accuracy"] is None:
        return None
    present = [metrics[c]["accuracy"] for c in group if c in metrics and metrics[c]["accuracy"] is not None]
    if not present:
        return None
    return metrics[CANAL_CONDITION]["accuracy"] - (sum(present) / len(present))


@torch.no_grad()
def weighted_nll(model, loader, device, class_weights):
    """Mean weighted cross-entropy, no label smoothing -- matches train.py's
    'sanity-check criterion' rather than its smoothed training criterion, so
    both checkpoints are compared on the same footing."""
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        reduction="mean",
    )
    total_loss, n_batches = 0.0, 0
    for tensors, labels, study_ids, conditions, levels in loader:
        tensors, labels = tensors.to(device), labels.to(device)
        logits = model(tensors)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / n_batches if n_batches else None


def classify_delta(new_val, old_val, higher_is_better, band):
    """'IMPROVED' / 'HELD' / 'WORSE', or None if either value is missing."""
    if new_val is None or old_val is None:
        return None
    delta = (new_val - old_val) if higher_is_better else (old_val - new_val)
    if delta > band:
        return "IMPROVED"
    if delta < -band:
        return "WORSE"
    return "HELD"


def cm_to_jsonable(metrics_dict):
    return {
        c: {k: (v.tolist() if k == "cm" else v) for k, v in m.items()}
        for c, m in metrics_dict.items()
    }


def print_checkpoint_block(label, metrics, mod_recall, mod_n, pooled_acc, gap_all, gap_subart, gap_foram, nll):
    print(f"\n--- {label} ---")
    print(f"  pooled accuracy          = {pooled_acc:.4f}")
    print(f"  pooled Moderate recall   = {mod_recall:.4f}  (n={mod_n})")
    print(f"  lateralized gap (4-cond) = {gap_all:+.4f}")
    print(f"    gap vs subarticular    = {gap_subart:+.4f}")
    print(f"    gap vs foraminal       = {gap_foram:+.4f}")
    print(f"  weighted NLL             = {nll:.4f}")
    print(f"  per-condition accuracy / Moderate recall:")
    for cond in sorted(metrics.keys()):
        m = metrics[cond]
        mod_r = m["recalls"][MODERATE_CLASS_IDX]
        mod_r_str = f"{mod_r:.4f}" if mod_r is not None else "n/a"
        print(f"    {cond:35s} n={m['n']:5d}  acc={m['accuracy']:.4f}  moderate_recall={mod_r_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--expect-new-sha-prefix", required=True)
    parser.add_argument("--baseline-checkpoint", default=None,
                         help="Pre-retrain checkpoint blob extracted from git history. "
                              "Omit if no earlier version exists -- verdict becomes NO_EXACT_BASELINE.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--noise-band", type=float, default=0.03,
                         help="+/- band on new-old before a metric counts as IMPROVED/WORSE vs HELD.")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify_retrain_delta] device={device}")

    # --- integrity checks ---------------------------------------------------
    new_sha = sha256_of_file(args.new_checkpoint)
    print(f"[verify_retrain_delta] new checkpoint sha256={new_sha}")
    if not new_sha.startswith(args.expect_new_sha_prefix):
        print(f"HARD STOP: new checkpoint sha256 does not start with "
              f"'{args.expect_new_sha_prefix}' (got '{new_sha[:12]}...'). "
              f"Wrong file staged at --new-checkpoint -- fix the path before continuing.")
        sys.exit(1)

    have_baseline = args.baseline_checkpoint is not None
    old_sha = None
    if have_baseline:
        old_sha = sha256_of_file(args.baseline_checkpoint)
        print(f"[verify_retrain_delta] baseline checkpoint sha256={old_sha}")
        if old_sha == new_sha:
            print("HARD STOP: baseline blob is byte-identical to the new checkpoint. "
                  "Step 3 extracted the wrong commit -- redo it.")
            sys.exit(1)

    # --- data -----------------------------------------------------------------
    loaders = build_classification_dataloaders(args.manifest, batch_size=args.batch_size)
    print(f"[verify_retrain_delta] build_classification_dataloaders returned {len(loaders)} item(s)")
    _, val_loader = loaders  # mirrors verify_training_quality.py's own unpacking

    # --- new checkpoint ---------------------------------------------------
    new_model = load_model(args.new_checkpoint, device)
    new_by_condition = collect_predictions(new_model, val_loader, device)

    missing = ALL_CONDITIONS - set(new_by_condition.keys())
    if missing:
        print(f"HARD STOP: val predictions are missing condition(s): {sorted(missing)}. "
              f"Cannot compute the lateralized gap or pooled Moderate recall without all 5.")
        sys.exit(1)

    total_val_rows = sum(len(y) for y, _ in new_by_condition.values())
    lo, hi = EXPECTED_VAL_ROW_RANGE
    if not (lo <= total_val_rows <= hi):
        print(f"WARNING: total val rows = {total_val_rows}, outside documented range "
              f"[{lo}, {hi}]. Confirm --manifest is the post-migration manifest train.py used.")

    new_metrics = per_condition_metrics(new_by_condition)
    new_mod_recall, new_mod_n, new_pooled_acc = pooled_moderate_recall(new_by_condition)
    new_gap_all = lateralized_gap(new_metrics, LATERALIZED_CONDITIONS)
    new_gap_subart = lateralized_gap(new_metrics, SUBARTICULAR_CONDITIONS)
    new_gap_foram = lateralized_gap(new_metrics, FORAMINAL_CONDITIONS)
    new_nll = weighted_nll(new_model, val_loader, device, CLASS_WEIGHTS)

    canal_acc = new_metrics[CANAL_CONDITION]["accuracy"]
    clo, chi = DOCUMENTED_CANAL_ACC_RANGE
    if canal_acc is not None and not (clo <= canal_acc <= chi):
        print(f"WARNING: new checkpoint's canal accuracy ({canal_acc:.4f}) is outside the "
              f"documented 0.86-0.89 +/-0.02 band. Canal crops weren't touched by the migration -- "
              f"a shift here usually means something else changed (wrong checkpoint, wrong "
              f"manifest, wrong val split).")

    result = {
        "new_checkpoint": {
            "path": args.new_checkpoint,
            "sha256": new_sha,
            "total_val_rows": total_val_rows,
            "pooled_accuracy": new_pooled_acc,
            "pooled_moderate_recall": new_mod_recall,
            "pooled_moderate_n": new_mod_n,
            "lateralized_gap_all_4": new_gap_all,
            "gap_vs_subarticular": new_gap_subart,
            "gap_vs_foraminal": new_gap_foram,
            "weighted_nll": new_nll,
            "per_condition": cm_to_jsonable(new_metrics),
        },
        "baseline_checkpoint": None,
        "verdict": None,
        "noise_band": args.noise_band,
    }

    # --- baseline checkpoint, same rows -----------------------------------
    old_metrics = old_mod_recall = old_gap_all = old_gap_subart = old_gap_foram = old_nll = old_pooled_acc = None
    if have_baseline:
        old_model = load_model(args.baseline_checkpoint, device)
        old_by_condition = collect_predictions(old_model, val_loader, device)
        missing_old = ALL_CONDITIONS - set(old_by_condition.keys())
        if missing_old:
            print(f"HARD STOP: baseline val predictions are missing condition(s): {sorted(missing_old)}.")
            sys.exit(1)

        old_metrics = per_condition_metrics(old_by_condition)
        old_mod_recall, old_mod_n, old_pooled_acc = pooled_moderate_recall(old_by_condition)
        old_gap_all = lateralized_gap(old_metrics, LATERALIZED_CONDITIONS)
        old_gap_subart = lateralized_gap(old_metrics, SUBARTICULAR_CONDITIONS)
        old_gap_foram = lateralized_gap(old_metrics, FORAMINAL_CONDITIONS)
        old_nll = weighted_nll(old_model, val_loader, device, CLASS_WEIGHTS)

        old_canal_acc = old_metrics[CANAL_CONDITION]["accuracy"]
        invalid_baseline = old_canal_acc is not None and not (clo <= old_canal_acc <= chi)
        if invalid_baseline:
            print(f"WARNING: baseline checkpoint's canal accuracy ({old_canal_acc:.4f}) is outside "
                  f"the documented 0.86-0.89 +/-0.02 band. This is a backstop, not proof of the "
                  f"right blob -- re-check the commit hash/date/message used to extract it.")

        result["baseline_checkpoint"] = {
            "path": args.baseline_checkpoint,
            "sha256": old_sha,
            "pooled_accuracy": old_pooled_acc,
            "pooled_moderate_recall": old_mod_recall,
            "pooled_moderate_n": old_mod_n,
            "lateralized_gap_all_4": old_gap_all,
            "gap_vs_subarticular": old_gap_subart,
            "gap_vs_foraminal": old_gap_foram,
            "weighted_nll": old_nll,
            "per_condition": cm_to_jsonable(old_metrics),
        }

        if invalid_baseline:
            verdict = "INVALID_BASELINE"
        else:
            mod_delta = classify_delta(new_mod_recall, old_mod_recall, higher_is_better=True, band=args.noise_band)
            gap_delta = classify_delta(new_gap_all, old_gap_all, higher_is_better=False, band=args.noise_band)
            deltas = [d for d in (mod_delta, gap_delta) if d is not None]
            if "WORSE" in deltas:
                verdict = "NOT_IMPROVED"
            elif "IMPROVED" in deltas:
                verdict = "IMPROVED"
            else:
                verdict = "HELD"
        result["verdict"] = verdict
    else:
        result["verdict"] = "NO_EXACT_BASELINE"

    # --- report -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE C RESULT")
    print("=" * 70)
    print_checkpoint_block("NEW (retrained)", new_metrics, new_mod_recall, new_mod_n,
                            new_pooled_acc, new_gap_all, new_gap_subart, new_gap_foram, new_nll)
    if have_baseline:
        print_checkpoint_block("BASELINE (pre-retrain)", old_metrics, old_mod_recall, old_mod_n,
                                old_pooled_acc, old_gap_all, old_gap_subart, old_gap_foram, old_nll)
    print(f"\nVERDICT: {result['verdict']}")
    if result["verdict"] == "NOT_IMPROVED":
        print("Recorded. Per plan, this routes to the deferred 32/10 skew investigation -- "
              "that investigation is NOT opened by this script. Treat this as a stop point.")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n[verify_retrain_delta] wrote {args.out_json}")


if __name__ == "__main__":
    main()

"""
classification/verify_training_quality.py

Standalone check for "is the trained checkpoint actually sane, per condition"
-- specifically built to answer: is axial subarticular-stenosis collapsing
relative to the other four conditions, per the concern raised after the
localization mAP diagnostic (pdscd_progress_update_v2.md §3).

Does NOT retrain anything. Loads the saved checkpoint, runs one pass over
the val_loader already built by build_classification_dataloaders(), and
reports:
  - per-condition accuracy AND a full 3x3 confusion matrix (accuracy alone
    hides a model that always predicts "Normal/Mild" and looks fine on a
    class-imbalanced set)
  - per-condition macro-F1 (imbalance-robust, unlike raw accuracy)
  - explicit flag if axial-derived conditions (left/right subarticular
    stenosis) macro-F1 trails sagittal-derived conditions by more than a
    configurable margin
  - explicit flag if any single class within any condition has recall < 0.15
    (a near-total miss on Severe, e.g., would be exactly the kind of failure
    an aggregate accuracy number hides)

Run this BEFORE deciding train.py's result is "stable enough" to build on.
"""
import sys
import argparse
from collections import defaultdict

import torch
import numpy as np
from torchvision.models import efficientnet_b3

sys.path.append("/kaggle/working/PDSCD")
from classification.build_dataloaders import build_classification_dataloaders

SEVERITY_LABELS = ["Normal/Mild", "Moderate", "Severe"]
AXIAL_CONDITIONS = {"left_subarticular_stenosis", "right_subarticular_stenosis"}
SAGITTAL_CONDITIONS = {
    "spinal_canal_stenosis",
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
}

AXIAL_GAP_FLAG_THRESHOLD = 0.10   # macro-F1 points
MIN_ACCEPTABLE_RECALL = 0.15      # per-class recall floor


def load_model(checkpoint_path, device):
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, 3)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Returns dict: condition -> (y_true list, y_pred list)"""
    by_condition = defaultdict(lambda: ([], []))
    for tensors, labels, study_ids, conditions, levels in loader:
        tensors = tensors.to(device)
        logits = model(tensors)
        preds = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.numpy()
        for i, cond in enumerate(conditions):
            by_condition[cond][0].append(int(labels_np[i]))
            by_condition[cond][1].append(int(preds[i]))
    return by_condition


def confusion_matrix_3x3(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_recall(cm):
    recalls = []
    for c in range(3):
        row_sum = cm[c].sum()
        recalls.append(cm[c, c] / row_sum if row_sum > 0 else None)
    return recalls


def macro_f1(cm):
    f1s = []
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / 3


def print_report(by_condition):
    print("\n" + "=" * 70)
    print("PER-CONDITION BREAKDOWN")
    print("=" * 70)

    macro_f1_by_condition = {}
    for cond in sorted(by_condition.keys()):
        y_true, y_pred = by_condition[cond]
        cm = confusion_matrix_3x3(y_true, y_pred)
        acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
        f1 = macro_f1(cm)
        recalls = per_class_recall(cm)
        macro_f1_by_condition[cond] = f1

        print(f"\n{cond}  (n={cm.sum()})")
        print(f"  accuracy={acc:.4f}  macro_f1={f1:.4f}")
        print(f"  confusion matrix (rows=true, cols=pred), "
              f"order={SEVERITY_LABELS}")
        for r in range(3):
            print(f"    {SEVERITY_LABELS[r]:12s}  {cm[r].tolist()}")
        for c in range(3):
            r = recalls[c]
            flag = "  <-- LOW RECALL" if (r is not None and r < MIN_ACCEPTABLE_RECALL) else ""
            r_str = f"{r:.4f}" if r is not None else "n/a (0 samples)"
            print(f"    recall[{SEVERITY_LABELS[c]}] = {r_str}{flag}")

    print("\n" + "=" * 70)
    print("AXIAL vs SAGITTAL CHECK (the specific concern being verified)")
    print("=" * 70)
    axial_f1s = [macro_f1_by_condition[c] for c in AXIAL_CONDITIONS if c in macro_f1_by_condition]
    sagittal_f1s = [macro_f1_by_condition[c] for c in SAGITTAL_CONDITIONS if c in macro_f1_by_condition]

    if not axial_f1s or not sagittal_f1s:
        print("  Could not compute -- one of the two groups had no predictions in this run.")
        return

    axial_mean = sum(axial_f1s) / len(axial_f1s)
    sagittal_mean = sum(sagittal_f1s) / len(sagittal_f1s)
    gap = sagittal_mean - axial_mean

    print(f"  axial (subarticular) mean macro_f1:    {axial_mean:.4f}")
    print(f"  sagittal (other 3) mean macro_f1:       {sagittal_mean:.4f}")
    print(f"  gap (sagittal - axial):                 {gap:+.4f}")

    if gap > AXIAL_GAP_FLAG_THRESHOLD:
        print(f"  ⚠ FLAG: axial trails sagittal by more than {AXIAL_GAP_FLAG_THRESHOLD:.2f} "
              f"macro-F1 points. This is the exact axial-specific weakness the earlier "
              f"localization mAP diagnostic looked for and did not find at the detection "
              f"stage -- worth checking whether it shows up here at the classification "
              f"stage instead, before treating the checkpoint as stable.")
    else:
        print(f"  OK -- no axial-specific collapse relative to the {AXIAL_GAP_FLAG_THRESHOLD:.2f} "
              f"flag threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/kaggle/working/checkpoints/efficientnet_b3_best.pt")
    parser.add_argument("--manifest", required=True,
                         help="Path to manifest.csv, same one train.py used.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify] device={device}")
    print(f"[verify] loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    _, val_loader = build_classification_dataloaders(args.manifest, batch_size=args.batch_size)
    by_condition = collect_predictions(model, val_loader, device)
    print_report(by_condition)


if __name__ == "__main__":
    main()

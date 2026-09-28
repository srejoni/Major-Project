"""
Ensemble-level recheck of subarticular Moderate recall (L/R).

Why
---
Phase C found subarticular Moderate recall regressed on EfficientNet-B3
after the laterality-fix crop dataset landed (L -0.0665, R -0.0836), while
canal/foraminal improved. ConvNeXt-Tiny and ResNet34 were trained on the
same post-fix crops, so the regression likely carries into them too -- but
it has never been checked at the *ensemble* (soft-averaged, all 3 archs)
level. This is completion_plan.md, "Remaining to finish line", item 4 --
the only imaging-owned task left before the finish line.

Run this in the same env you trained in (Kaggle kernel, repo cloned,
checkpoints LFS-pulled). Project convention: absolute + quoted paths,
confirm paths with !ls / !find before running.

ADAPT blocks: I don't have crop_dataset.py / schema.py in this session, so
I can't know your dataloader's exact batch keys or the exact Condition
enum strings. Fill those two blocks in from your own source before running
-- everything else should work once they're plugged in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, recall_score

# --- your existing modules -------------------------------------------------
from ensemble_predict import CHECKPOINT_REGISTRY, load_models, soft_probability_averaging
from build_dataloaders import build_classification_dataloaders  # adjust path if it lives elsewhere

SEVERITY_LABELS = ["Normal/Mild", "Moderate", "Severe"]  # imaging_context.md §1
SUBARTICULAR_CONDITIONS = {
    "left": "left_subarticular_stenosis",    # ADAPT: match schema.Condition exactly
    "right": "right_subarticular_stenosis",  # ADAPT: match schema.Condition exactly
}
# Phase C single-model (EfficientNet-B3) deltas, imaging_context.md §8
PHASE_C_DELTA = {"left": -0.0665, "right": -0.0836}


def preflight_check_models(models: dict) -> None:
    """Guard against the past incident where resnet34 silently loaded
    2-of-3 models with no error (imaging_context.md §8 housekeeping)."""
    expected = set(CHECKPOINT_REGISTRY.keys())
    got = set(models.keys())
    assert got == expected, (
        f"Ensemble is missing models -- expected {expected}, loaded {got}. "
        "Do not trust any recall numbers from a partial ensemble."
    )
    print(f"[preflight] {len(models)} models loaded: {sorted(got)}")


@torch.no_grad()
def run_ensemble_inference(dataloader, models: dict, device: str) -> pd.DataFrame:
    """Run all 3 models over one split, soft-average per the locked
    ensemble method, and return one row per crop."""
    for m in models.values():
        m.eval()

    rows = []
    for batch in dataloader:
        # ---------------------------------------------------------- ADAPT
        images = batch["image"].to(device)
        study_ids = batch["study_id"]
        conditions = batch["condition"]        # e.g. "left_subarticular_stenosis"
        levels = batch["level"]                # e.g. "L4/L5"
        severities = batch["severity_label"]   # ground truth, e.g. "Moderate"
        # -------------------------------------------------------- /ADAPT

        per_model_probs = {}
        for arch_name, model in models.items():
            logits = model(images)
            per_model_probs[arch_name] = torch.softmax(logits, dim=1).cpu().numpy()

        ensemble_probs = soft_probability_averaging(per_model_probs)
        ensemble_pred = ensemble_probs.argmax(axis=1)

        for i in range(len(study_ids)):
            row = {
                "study_id": study_ids[i],
                "condition": conditions[i],
                "level": levels[i],
                "severity_true": severities[i],
                "pred_ensemble": SEVERITY_LABELS[int(ensemble_pred[i])],
            }
            for arch_name, probs in per_model_probs.items():
                row[f"pred_{arch_name}"] = SEVERITY_LABELS[int(probs[i].argmax())]
            rows.append(row)

    return pd.DataFrame(rows)


def moderate_recall(df: pd.DataFrame, condition_value: str, pred_col: str) -> float:
    sub = df[df["condition"] == condition_value]
    if sub.empty:
        raise ValueError(
            f"No rows for condition={condition_value!r} -- check the ADAPT block / your Condition enum spelling."
        )
    return recall_score(sub["severity_true"], sub[pred_col], labels=["Moderate"], average="micro", zero_division=0)


def summarize(df: pd.DataFrame) -> dict:
    """Moderate recall, per side, per prediction source (ensemble + each arch)."""
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    return {
        side: {col.replace("pred_", ""): round(moderate_recall(df, cond, col), 4) for col in pred_cols}
        for side, cond in SUBARTICULAR_CONDITIONS.items()
    }


def confusion_by_side(df: pd.DataFrame, pred_col: str = "pred_ensemble") -> dict[str, pd.DataFrame]:
    """Where Moderate errors go (e.g. systematically downgraded to Normal/Mild)."""
    out = {}
    for side, cond in SUBARTICULAR_CONDITIONS.items():
        sub = df[df["condition"] == cond]
        cm = confusion_matrix(sub["severity_true"], sub[pred_col], labels=SEVERITY_LABELS)
        out[side] = pd.DataFrame(
            cm, index=[f"true_{s}" for s in SEVERITY_LABELS], columns=[f"pred_{s}" for s in SEVERITY_LABELS]
        )
    return out


def decide(summary: dict, baseline: dict | None) -> str:
    lines = ["", "=== Ensemble recheck: subarticular Moderate recall ===", ""]
    for side in ("left", "right"):
        lines.append(f"{side.upper()}:")
        for source, recall in summary[side].items():
            lines.append(f"  {source:16s} recall(Moderate) = {recall:.4f}")
    lines.append("")

    if baseline is None:
        lines.append(
            "No pre-Phase-C baseline supplied -- pass --baseline-left/--baseline-right "
            "(your EfficientNet-B3 Moderate recall numbers from before the laterality-fix "
            "crop dataset) for an automatic verdict. Otherwise compare the numbers above to "
            "your own Phase C logs by hand."
        )
        return "\n".join(lines)

    ok = []
    for side in ("left", "right"):
        current_delta = summary[side]["ensemble"] - baseline[side]
        recorded_delta = PHASE_C_DELTA[side]
        worse = current_delta < recorded_delta
        ok.append(not worse)
        lines.append(
            f"{side.upper()}: ensemble delta vs pre-Phase-C baseline = {current_delta:+.4f} "
            f"(Phase C single-model delta was {recorded_delta:+.4f}) -> "
            f"{'WORSE than Phase C' if worse else 'no worse than Phase C'}"
        )

    lines.append("")
    if all(ok):
        lines.append(
            "RECOMMENDATION: ACCEPT as documented v1.0.0 limitation. Write it up and hand "
            "to your teammate for the Phase 8.4 limitations section -- no need to reopen imaging."
        )
    else:
        lines.append(
            "RECOMMENDATION: REOPEN imaging (completion_plan.md 'Deferred' items): row-loss "
            "skew diagnostic, GT-swap overlap check (incl. study 52695609), and the 22-study "
            "exclude/flag decision. If this changes crops/labels, retrain, rerun "
            "verify_phase_f_export.py, then re-export and redeliver to your teammate."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True, help="absolute, quoted path to the crop manifest CSV")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline-left", type=float, default=None)
    parser.add_argument("--baseline-right", type=float, default=None)
    parser.add_argument("--out-dir", default="handoff/phase_i_recheck")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(args.device)
    preflight_check_models(models)

    _, val_loader, test_loader = build_classification_dataloaders(args.manifest_path, batch_size=16, num_workers=2)
    loader = val_loader if args.split == "val" else test_loader

    df = run_ensemble_inference(loader, models, args.device)
    df.to_csv(out_dir / f"subarticular_recheck_{args.split}_raw.csv", index=False)

    summary = summarize(df)
    (out_dir / f"subarticular_recheck_{args.split}_summary.json").write_text(json.dumps(summary, indent=2))

    for side, cm in confusion_by_side(df).items():
        cm.to_csv(out_dir / f"subarticular_recheck_{args.split}_confusion_{side}.csv")

    baseline = None
    if args.baseline_left is not None and args.baseline_right is not None:
        baseline = {"left": args.baseline_left, "right": args.baseline_right}

    report = decide(summary, baseline)
    print(report)
    (out_dir / f"subarticular_recheck_{args.split}_report.txt").write_text(report)


if __name__ == "__main__":
    main()

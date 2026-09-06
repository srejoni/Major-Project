"""
classification/train.py

Training loop for the basic model, fixing the per-target class-weighting
issue, PLUS two throughput accelerations for this T4x2 environment:
  1. nn.DataParallel across both GPUs when more than one is available --
     the session was observed running at 94%/0% GPU utilization on the two
     T4s, meaning only one was ever actually used.
  2. Automatic mixed precision (torch.cuda.amp) -- T4s have strong FP16
     tensor cores; this was previously running in full FP32.

Neither change affects the class-weighting fix, the per-condition eval
breakdown, or the loss-weighting sanity check below -- all unchanged from
the previous version.

Model architecture note: no classification/model.py has been shared into
this conversation, so this script builds a standard EfficientNetB3
(torchvision, ImageNet-pretrained) with its final classifier layer replaced
by Linear(in_features, 3), per the project doc's stated "EfficientNetB3 +
head" plan. If an existing model-definition file already differs from
this, swap the import in build_model() below for that instead -- flagging
this assumption explicitly rather than silently guessing it matches.
"""

import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

sys.path.append("/kaggle/working/PDSCD")
from classification.build_dataloaders import build_classification_dataloaders

# Confirmed mount from the crop-extraction dataset check -- update if this
# ever moves.
MANIFEST_PATH = "/kaggle/input/datasets/srejoni/crop-extraction/classification_data/manifest.csv"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# Fixed competition severity weights. class idx 0/1/2 = normal-mild/
# moderate/severe. Applied identically to every batch, regardless of which
# condition/level the batch's samples belong to -- this is NOT a per-target
# class-balancing weight, and NOT derived from the 254/741/980 patient-level
# stratification distribution.
SEVERITY_CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
SEVERITY_LABELS = ["Normal/Mild", "Moderate", "Severe"]


def build_model(device):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 3)
    model = model.to(device)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"[train] {n_gpus} GPUs detected -- wrapping model in nn.DataParallel")
        model = nn.DataParallel(model)
    else:
        print(f"[train] {n_gpus} GPU(s) detected -- running without DataParallel")
    return model


def unwrap_model(model):
    """DataParallel wraps the real model under .module -- saving
    state_dict() on the wrapper itself puts a 'module.' prefix on every key,
    which silently breaks loading that checkpoint into a plain (non-wrapped)
    model later (inference, or a differently-configured ensemble run).
    Always save/load through the unwrapped model to avoid that mismatch."""
    return model.module if isinstance(model, nn.DataParallel) else model


def verify_loss_weighting(device):
    """Runtime check, not an assumption: confirms CrossEntropyLoss(reduction=
    'mean') behaves as already hand-validated in isolation (normalizes by
    sum-of-weights-in-batch, not sample count) before this script existed.
    This is a cheap smoke test that fails loudly if the environment/PyTorch
    version ever changes that behavior -- it does not re-derive the original
    validation, which already happened separately."""
    weights = SEVERITY_CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="mean")

    logits = torch.zeros(3, 3, device=device)
    logits[0, 0] = 10.0   # confident, correct, class 0 (weight 1)
    logits[1, 1] = 10.0   # confident, correct, class 1 (weight 2)
    logits[2, 2] = 10.0   # confident, correct, class 2 (weight 4)
    targets = torch.tensor([0, 1, 2], device=device)

    loss = criterion(logits, targets).item()
    assert loss < 1e-3, (
        f"CrossEntropyLoss sanity check failed: expected near-zero loss for "
        f"confident/correct predictions, got {loss:.4f}. Do not proceed until "
        f"this is understood."
    )
    print(f"[train] loss-weighting sanity check passed (loss={loss:.6f})")
    return criterion


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, n_batches = 0.0, 0
    for tensors, labels, study_ids, conditions, levels in loader:
        tensors, labels = tensors.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(tensors)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns overall val loss/accuracy plus a per-condition breakdown --
    this is what lets a series-specific weakness (e.g. axial vs. sagittal,
    per the localization mAP concern flagged earlier) show up during
    training instead of staying hidden inside one aggregate number. This
    breakdown is diagnostic only; it does not change the loss weighting.
    Forward passes also run under autocast for the same throughput benefit;
    no scaler is needed here since there's no backward pass."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct, total = 0, 0
    per_condition = defaultdict(lambda: {"loss": 0.0, "n": 0, "correct": 0})

    for tensors, labels, study_ids, conditions, levels in loader:
        tensors, labels = tensors.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            logits = model(tensors)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i, cond in enumerate(conditions):
            per_condition[cond]["loss"] += loss.item()
            per_condition[cond]["n"] += 1
            per_condition[cond]["correct"] += int(preds[i].item() == labels[i].item())

    val_loss = total_loss / max(n_batches, 1)
    val_acc = correct / max(total, 1)

    print(f"[eval] val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    for cond, stats in sorted(per_condition.items()):
        cond_acc = stats["correct"] / max(stats["n"], 1)
        print(f"  {cond:35s}  n={stats['n']:5d}  acc={cond_acc:.4f}")

    return val_loss, val_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    criterion = verify_loss_weighting(device)
    train_loader, val_loader = build_classification_dataloaders(
        MANIFEST_PATH, batch_size=BATCH_SIZE
    )

    model = build_model(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        print(f"[epoch {epoch}/{NUM_EPOCHS}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(CHECKPOINT_DIR, "efficientnet_b3_best.pt")
            torch.save(unwrap_model(model).state_dict(), ckpt_path)
            print(f"[train] new best val_loss={val_loss:.4f} -- saved {ckpt_path}")


if __name__ == "__main__":
    main()

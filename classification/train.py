"""
classification/train.py

Training loop for the basic model. Adds overfitting countermeasures on top
of the existing class-weighting fix and DataParallel/AMP acceleration:

  1. Weight decay (AdamW's built-in L2 penalty) -- discourages large
     weights, directly targets memorization.
  2. Label smoothing (0.1) on the training loss -- softens hard 0/1
     targets so the model isn't rewarded for overconfident wrong-direction
     predictions. IMPORTANT: this is applied via a SEPARATE criterion from
     the one used in verify_loss_weighting()'s sanity check below. Label
     smoothing makes even a perfect, fully-confident prediction produce a
     small non-zero loss (because the smoothed target is never a true
     one-hot), so running the sanity check's near-zero-loss assertion
     against a label-smoothed criterion would fail for the wrong reason
     and mask a real bug if one ever occurred. The sanity check keeps
     validating weighting mechanics only, exactly as before; smoothing is
     layered on afterward, at the actual training criterion, deliberately
     kept as two objects.
  3. ReduceLROnPlateau scheduler keyed on val_loss -- previously a fixed
     LR for all 10 epochs. This was very likely a real contributor to the
     val_loss oscillation observed (0.548 -> 0.577 -> 0.568 -> 0.592 -> ...)
     rather than a smooth climb: a fixed LR that was fine for early-epoch
     progress can be too large once the loss surface flattens out, causing
     it to bounce around a minimum instead of settling into it.
  4. Early stopping (patience=3 epochs on val_loss) -- stops the run once
     val_loss hasn't improved on the best-seen value for 3 consecutive
     epochs, instead of always running the full fixed NUM_EPOCHS. The
     `if val_loss < best_val_loss` checkpoint logic is unchanged and still
     the source of truth for which weights get saved.

Neither DataParallel, AMP, the class-weighting fix, nor the per-condition
eval breakdown are touched -- all unchanged from the previous version.

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 3  # epochs with no val_loss improvement before stopping
LR_SCHEDULER_PATIENCE = 1    # epochs with no val_loss improvement before LR is halved
LR_SCHEDULER_FACTOR = 0.5

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
    validation, which already happened separately.

    Deliberately uses NO label smoothing here, even though the actual
    training criterion (built in main()) does. Label smoothing changes what
    "near-zero loss for a confident correct prediction" means -- with
    smoothing on, even a perfect prediction has a non-zero floor loss from
    the smoothed target distribution. Testing weighting mechanics and
    testing smoothing behavior are two different concerns; keeping this
    criterion unsmoothed means a real weighting regression still fails
    loudly and specifically, instead of being masked by or confused with
    smoothing's expected non-zero floor."""
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


def build_training_criterion(device):
    """The actual criterion used for backprop -- same class weights as the
    sanity-checked one above, plus label smoothing layered on top. Kept as
    a separate object from verify_loss_weighting()'s criterion (see that
    function's docstring for why)."""
    weights = SEVERITY_CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights, reduction="mean", label_smoothing=LABEL_SMOOTHING
    )
    print(f"[train] training criterion built with label_smoothing={LABEL_SMOOTHING}")
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
    no scaler is needed here since there's no backward pass.

    NOTE: evaluate() uses the SAME (label-smoothed) criterion as training,
    intentionally -- val_loss must be measured on the same loss surface the
    optimizer is actually descending, or "best val_loss" checkpointing
    would be comparing numbers that aren't apples-to-apples with what
    training is optimizing against."""
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

    # Sanity check runs on its own unsmoothed criterion -- see docstring.
    verify_loss_weighting(device)
    # Actual training/eval criterion, with label smoothing.
    criterion = build_training_criterion(device)

    train_loader, val_loader = build_classification_dataloaders(
        MANIFEST_PATH, batch_size=BATCH_SIZE
    )

    model = build_model(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )
    scaler = torch.cuda.amp.GradScaler()

    print(f"[train] weight_decay={WEIGHT_DECAY}  "
          f"early_stopping_patience={EARLY_STOPPING_PATIENCE}  "
          f"lr_scheduler_patience={LR_SCHEDULER_PATIENCE}  "
          f"lr_scheduler_factor={LR_SCHEDULER_FACTOR}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch}/{NUM_EPOCHS}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={current_lr:.2e}  "
              f"({elapsed:.1f}s)")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            ckpt_path = os.path.join(CHECKPOINT_DIR, "efficientnet_b3_best.pt")
            torch.save(unwrap_model(model).state_dict(), ckpt_path)
            print(f"[train] new best val_loss={val_loss:.4f} -- saved {ckpt_path}")
        else:
            epochs_without_improvement += 1
            print(f"[train] no improvement for {epochs_without_improvement} "
                  f"epoch(s) (best={best_val_loss:.4f})")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"[train] early stopping -- no val_loss improvement for "
                      f"{EARLY_STOPPING_PATIENCE} consecutive epochs. Best "
                      f"checkpoint (val_loss={best_val_loss:.4f}) remains at "
                      f"{os.path.join(CHECKPOINT_DIR, 'efficientnet_b3_best.pt')}.")
                break

    print(f"[train] training complete. best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()

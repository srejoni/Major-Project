"""
classification/model_registry.py

Single source of truth for "how do I build a PDSCD classification model
for a given architecture" -- fixes the Phase D blocker in progress log §7
/ next-actions step 4: build_model() in train.py only ever constructed
efficientnet_b3, and the checkpoint filename was hardcoded to
"efficientnet_b3_best.pt" in two places (the save line and the
early-stopping log message).

Does NOT change the fine-tuning setup. Optimizer, scheduler, label
smoothing, augmentation, early stopping, DataParallel/AMP/unwrap_model()
all still live in train.py and already operate on model.parameters() /
whatever this module returns -- per the locked decision (progress log §6:
identical full fine-tune across all 3 architectures), none of that needed
to change. Only architecture construction and the checkpoint filename
were architecture-specific; both are centralized here. Device placement
and DataParallel wrapping stay in train.py's build_model(device,
model_name), since neither is architecture-specific either.

Weight enums are pinned explicitly (IMAGENET1K_V1 per architecture) to
match train.py's existing EfficientNet_B3_Weights.IMAGENET1K_V1 style,
rather than torchvision's generic "DEFAULT" alias.
"""
from torch import nn
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    resnet34, ResNet34_Weights,
)

NUM_CLASSES = 3  # Normal/Mild, Moderate, Severe

# Where each architecture's classification head lives, and how to reach it.
# efficientnet_b3 / convnext_tiny: `.classifier` is a Sequential ending in
#   nn.Linear -- swap classifier[-1] (exactly what train.py's original
#   build_model() and verify_training_quality.py already do).
# resnet34: `.fc` IS the Linear layer directly -- not indexable, swap it whole.
MODEL_REGISTRY = {
    "efficientnet_b3": {
        "ctor": efficientnet_b3, "weights": EfficientNet_B3_Weights.IMAGENET1K_V1,
        "head_attr": "classifier", "head_is_sequential": True,
    },
    "convnext_tiny": {
        "ctor": convnext_tiny, "weights": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "head_attr": "classifier", "head_is_sequential": True,
    },
    "resnet34": {
        "ctor": resnet34, "weights": ResNet34_Weights.IMAGENET1K_V1,
        "head_attr": "fc", "head_is_sequential": False,
    },
}


def build_model(model_name, num_classes=NUM_CLASSES, pretrained=True):
    """Construct `model_name` with its classification head replaced for
    `num_classes`. Returned fully unfrozen -- nothing here sets
    requires_grad=False on anything, matching the locked full-fine-tune
    decision. `pretrained=True` downloads the pinned ImageNet weights
    (needs internet in the Kaggle session); set False for offline
    architecture checks, as the smoke test below does."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from {sorted(MODEL_REGISTRY)}.")
    spec = MODEL_REGISTRY[model_name]
    model = spec["ctor"](weights=spec["weights"] if pretrained else None)

    if spec["head_is_sequential"]:
        head = getattr(model, spec["head_attr"])
        in_features = head[-1].in_features
        head[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = getattr(model, spec["head_attr"]).in_features
        setattr(model, spec["head_attr"], nn.Linear(in_features, num_classes))

    return model


def checkpoint_filename(model_name):
    """Just the filename (not a full path), matching push_checkpoint.py /
    checkpoint_manifest.json's existing per-model-name convention -- drops
    straight into whatever CHECKPOINT_DIR-joining train.py already does."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from {sorted(MODEL_REGISTRY)}.")
    return f"{model_name}_best.pt"


if __name__ == "__main__":
    # Standalone smoke test. Run in your Kaggle session BEFORE touching
    # train.py or spending any GPU time:
    #   !python -m classification.model_registry
    # Confirms all 3 architectures construct, accept a real input tensor,
    # and emit shape (N, 3) with every parameter trainable.
    import torch

    for name in MODEL_REGISTRY:
        model = build_model(name, pretrained=False)  # skip the download for a quick offline check
        model.eval()
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        assert out.shape == (2, NUM_CLASSES), f"{name}: bad output shape {out.shape}"
        assert all(p.requires_grad for p in model.parameters()), \
            f"{name}: found frozen params -- violates the full-fine-tune decision"
        print(f"[model_registry] {name:16s} OK  output={tuple(out.shape)}  "
              f"params={n_params:,}  ckpt_filename={checkpoint_filename(name)}")
    print("[model_registry] all 3 architectures build correctly.")

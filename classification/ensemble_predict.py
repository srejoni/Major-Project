"""
classification/ensemble_predict.py

Runs the trained classification model(s) over a single crop and returns
per-architecture probabilities plus the ensemble average, shaped to match
pdscd_output_format.md's model_votes / model_agreement fields.

FIX (checkpoint path): MODEL_REGISTRY previously pointed at
"/kaggle/working/checkpoints/efficientnet_b3_best.pt" -- that's
train.py's CHECKPOINT_DIR, an ephemeral working-directory path from the
original training session that was never committed to git. The real,
committed checkpoint lives at classification/weights/efficientnet_b3_best.pt
(confirmed via `!ls /kaggle/working/PDSCD/classification/weights/`).
On a fresh clone, load_models() would raise FileNotFoundError at the old
path -- now fixed to point at the actual repo location, built from
REPO_ROOT the same way crop_extraction.py builds YOLO_WEIGHTS, instead of
introducing yet another standalone hardcoded absolute path.

FIX (fail loudly, with an actionable message): this project has now hit
this exact bug class three times in one conversation -- the competition
CSV path, the manifest.csv path, and now this checkpoint path, each time
a path that was valid in whichever session originally produced the file
but not in the repo as committed. load_models() now checks existence
before torch.load() and, if missing, prints the `!find` command that
resolves it -- matching the same pattern verify_paths.py already
established elsewhere in this project -- rather than surfacing a bare
FileNotFoundError from deep inside torch.

BLOCKER, NOT WORKED AROUND (unchanged): pdscd_output_schema.json requires
model_versions and model_votes to contain all three of efficientnet_b3,
convnext_tiny, resnet34 (additionalProperties: false). Only
efficientnet_b3 has a trained checkpoint anywhere in this project. This
file does NOT fabricate placeholder votes for the missing two to force a
schema pass -- copying the EfficientNetB3 result into convnext_tiny/
resnet34 slots would misrepresent single-model output as three-way
ensemble agreement, exactly the distinction model_agreement exists to
protect. Add convnext_tiny/resnet34 to MODEL_REGISTRY once real
checkpoints (and architecture-specific loaders, if needed) exist.
"""
import os

import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_b3

SEVERITY_KEYS = ["normal_mild", "moderate", "severe"]
SEVERITY_LABELS_DISPLAY = ["Normal/Mild", "Moderate", "Severe"]

# Same convention as crop_extraction.py's REPO_ROOT -- update both if this
# ever moves, but keep them the same constant in spirit rather than two
# independent hardcoded strings that can drift apart.
REPO_ROOT = "/kaggle/working/PDSCD"


def _load_efficientnet_b3(checkpoint_path, device):
    model = efficientnet_b3(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, 3)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


# Add convnext_tiny / resnet34 entries here once checkpoints exist. Each
# load_fn must return an eval()-mode model taking the same (3, 224, 224)
# normalized tensor crop_dataset.py already produces.
MODEL_REGISTRY = {
    "efficientnet_b3": {
        "checkpoint": f"{REPO_ROOT}/classification/weights/efficientnet_b3_best.pt",
        "load_fn": _load_efficientnet_b3,
    },
    # "convnext_tiny": {...},   # NOT YET TRAINED
    # "resnet34": {...},        # NOT YET TRAINED
}


def load_models(device):
    for name, spec in MODEL_REGISTRY.items():
        if not os.path.exists(spec["checkpoint"]):
            filename = os.path.basename(spec["checkpoint"])
            raise FileNotFoundError(
                f"Checkpoint for '{name}' not found at {spec['checkpoint']}. "
                f"This is the same class of stale-path bug this project has hit "
                f"before (competition CSVs, manifest.csv) -- don't guess a new "
                f"path, find the real one:\n"
                f"    !find /kaggle/working -iname \"{filename}\"\n"
                f"then update MODEL_REGISTRY['{name}']['checkpoint'] to match."
            )
    return {name: spec["load_fn"](spec["checkpoint"], device)
            for name, spec in MODEL_REGISTRY.items()}


@torch.no_grad()
def predict_crop(models, tensor, device):
    """tensor must already be normalized exactly as ClassificationCropDataset
    normalizes it -- see classification/normalization.py's normalize_crop(),
    the single shared implementation used by both training and inference.

    model_agreement is only computed (non-None) when exactly 3 models are
    present, because the schema's enum only defines unanimous_3_of_3 /
    majority_2_of_3 / split_no_majority -- there is no honest enum value
    for "2 models agreed" or "1 model voted." Returning a fabricated value
    for those cases would silently misrepresent confidence. Callers must
    treat a None agreement (from <3 models) as "not yet ensemble-ready,"
    not as a null they can ignore.
    """
    batch = tensor.unsqueeze(0).to(device)
    votes = {}
    prob_sum = torch.zeros(3, device=device)

    for name, model in models.items():
        logits = model(batch)[0]
        probs = F.softmax(logits, dim=0)
        severity_code = int(probs.argmax().item())
        votes[name] = {
            "severity_code": severity_code,
            "probabilities": {SEVERITY_KEYS[i]: round(probs[i].item(), 6) for i in range(3)},
        }
        prob_sum += probs

    n_models = len(models)
    avg_probs = (prob_sum / n_models).tolist()
    ensemble_code = int(torch.tensor(avg_probs).argmax().item())

    agreement = None
    if n_models == 3:
        codes = [v["severity_code"] for v in votes.values()]
        if len(set(codes)) == 1:
            agreement = "unanimous_3_of_3"
        elif max(codes.count(c) for c in set(codes)) == 2:
            agreement = "majority_2_of_3"
        else:
            agreement = "split_no_majority"

    return {
        "severity_code": ensemble_code,
        "severity_label": SEVERITY_LABELS_DISPLAY[ensemble_code],
        "confidence": round(max(avg_probs), 6),
        "probabilities": {SEVERITY_KEYS[i]: round(avg_probs[i], 6) for i in range(3)},
        "model_votes": votes,
        "model_agreement": agreement,
    }

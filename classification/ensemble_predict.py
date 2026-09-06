"""
classification/ensemble_predict.py

Runs the trained classification model(s) over a single crop and returns
per-architecture probabilities plus the ensemble average, shaped to match
pdscd_output_format.md's model_votes / model_agreement fields.

BLOCKER, NOT WORKED AROUND: pdscd_output_schema.json requires
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
import torch
import torch.nn.functional as F
from torchvision.models import efficientnet_b3

SEVERITY_KEYS = ["normal_mild", "moderate", "severe"]
SEVERITY_LABELS_DISPLAY = ["Normal/Mild", "Moderate", "Severe"]


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
        "checkpoint": "/kaggle/working/checkpoints/efficientnet_b3_best.pt",
        "load_fn": _load_efficientnet_b3,
    },
    # "convnext_tiny": {...},   # NOT YET TRAINED
    # "resnet34": {...},        # NOT YET TRAINED
}


def load_models(device):
    return {name: spec["load_fn"](spec["checkpoint"], device)
            for name, spec in MODEL_REGISTRY.items()}


@torch.no_grad()
def predict_crop(models, tensor, device):
    """tensor must already be normalized exactly as ClassificationCropDataset
    normalizes it -- see assemble_study_output.py's _normalize_crop(), kept
    numerically identical on purpose rather than reimplemented differently.

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

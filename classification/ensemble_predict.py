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

NOTE (checkpoint freshness, not a code bug -- a data/process one):
this file's correctness depends on each architecture's weights file
actually being the checkpoint from the latest completed train.py run
under all current fixes (merge-logic refactor + crop-root fix +
overfitting countermeasures). Because load_models() only checks that a
file exists at the registered path, not which training run produced it,
silently overwriting that path with a stale checkpoint would NOT raise
any error here -- it would just quietly serve outdated predictions.
Confirm the physical .pt file has actually been replaced and pushed
after every new training run, before trusting output from this module.

FIX (ConvNeXt-Tiny support): added a convnext_tiny MODEL_REGISTRY entry,
now that Phase D has produced a trained checkpoint.

FIX (stop duplicating architecture construction, this pass): this file
previously had its own _load_efficientnet_b3() / _load_convnext_tiny()
functions, each hand-rolling "build the torchvision model, read
in_features off the head, swap in a 3-class Linear, load the state dict."
That's exactly what classification/model_registry.py::build_model() now
does as the single source of truth (it handles both known head layouts --
efficientnet_b3/convnext_tiny's `.classifier[-1]` and resnet34's plain
`.fc` -- in one place, and train.py already depends on it for the same
reason). Reimplementing that logic here risked the two drifting apart
silently -- e.g. if model_registry.py's weights enum or head-swap logic
ever changed, this file would keep using the old version with no error.
Now imports build_model from there and only keeps what's genuinely
specific to this file: the checkpoint-path-per-architecture mapping and
turning that into loaded, eval()-mode models. Verified this refactor is
correct, not just plausible: model_registry.py's own smoke test
(`python -m classification.model_registry`) passed for all 3
architectures including resnet34's `.fc` branch before this file was
changed to depend on it.

Because architecture construction now comes from model_registry.py,
adding resnet34 here once it's trained is a one-line addition to
MODEL_REGISTRY below -- no new loader function needed, unlike before.

BLOCKER, NOT WORKED AROUND -- RESOLVED: pdscd_output_schema.json requires
model_versions and model_votes to contain all three of efficientnet_b3,
convnext_tiny, resnet34 (additionalProperties: false). This file never
fabricated placeholder votes for a missing architecture to force a schema
pass -- copying an existing result into an empty slot would misrepresent
partial-ensemble output as three-way agreement, exactly the distinction
model_agreement exists to protect.

FIX (resnet34 registered, this pass): resnet34's checkpoint was trained
and pushed (checkpoint_manifest.json confirms it, and Phase F's diagnostic
run confirmed the .pt file on disk loads cleanly via both torch.load and
build_classification_model("resnet34").load_state_dict()) but the
MODEL_REGISTRY entry below was left commented out with a stale "NOT YET
TRAINED" note -- the actual checkpoint's existence and this dict entry's
existence had silently drifted apart. load_models() had no way to detect
that drift: it only iterates whatever's in MODEL_REGISTRY, so a correctly
commented-out entry and an incorrectly-still-commented-out entry look
identical to it -- 2 models loaded, zero errors, no signal anything was
wrong until a downstream caller's separate 3-model guard caught it. Now
uncommented. All 3 architectures load; ensemble_predict.py's own
MODEL_REGISTRY is fully populated.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append("/kaggle/working/PDSCD")
from classification.model_registry import build_model as build_classification_model

SEVERITY_KEYS = ["normal_mild", "moderate", "severe"]
SEVERITY_LABELS_DISPLAY = ["Normal/Mild", "Moderate", "Severe"]

# Same convention as crop_extraction.py's REPO_ROOT -- update both if this
# ever moves, but keep them the same constant in spirit rather than two
# independent hardcoded strings that can drift apart.
REPO_ROOT = "/kaggle/working/PDSCD"

# Only the checkpoint-path-per-architecture mapping lives here now --
# architecture construction (which head attribute, sequential vs. plain,
# which pretrained-weights enum) is model_registry.py's job, not this
# file's. Add resnet34 here once it's trained: just another dict entry,
# no new function.
MODEL_REGISTRY = {
    "efficientnet_b3": {
        "checkpoint": f"{REPO_ROOT}/classification/weights/efficientnet_b3_best.pt",
    },
    "convnext_tiny": {
        "checkpoint": f"{REPO_ROOT}/classification/weights/convnext_tiny_best.pt",
    },
    "resnet34": {
        "checkpoint": f"{REPO_ROOT}/classification/weights/resnet34_best.pt",
    },
}


def _load_checkpoint(model_name, checkpoint_path, device):
    """pretrained=False deliberately -- we're about to overwrite every
    weight with the trained state_dict anyway, so downloading ImageNet
    weights first would only cost time/bandwidth and require network
    access this step doesn't need. Same pattern model_registry.py's own
    offline smoke test uses, for the same reason."""
    model = build_classification_model(model_name, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


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
    return {name: _load_checkpoint(name, spec["checkpoint"], device)
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

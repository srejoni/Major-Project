"""
classification/normalization.py

THE single normalization function for classification crops. Both training
(crop_dataset.py) and inference (assemble_study_output.py) must import and
call this -- never reimplement it. This project has already hit two
separate train/inference normalization-mismatch bugs (BOX_SIZE_FRAC,
percentile-vs-zscore); this file exists so a third can't happen here.

No functional defect found in the previous version -- the arithmetic here
was already numerically identical to what crop_dataset.py computed inline
before this file existed. What's added below is defensive hardening only:

- Explicit shape/dtype validation with an actionable error message,
  instead of relying on whatever cryptic error .permute(2, 0, 1) happens
  to throw if a future regression upstream ever hands this a malformed
  array (e.g. a (H, W) 2D crop missing its channel dimension).
- .contiguous() on the returned tensor. torch.from_numpy(...).permute(...)
  returns a view with non-standard strides; most PyTorch ops handle that
  transparently, but a few (certain custom ops, some serialization or
  C-extension code paths) expect contiguous memory and fail or silently
  copy anyway. Making it explicit and free here removes that as a future
  question mark.
"""
import numpy as np
import torch


def normalize_crop(crop_array: np.ndarray, mean: float, std: float) -> torch.Tensor:
    if not isinstance(crop_array, np.ndarray):
        raise TypeError(
            f"normalize_crop expected a numpy array, got {type(crop_array).__name__}. "
            f"Check the caller -- this usually means a raw path or PIL Image was "
            f"passed instead of the loaded array."
        )
    if crop_array.ndim != 3 or crop_array.shape[-1] != 3:
        raise ValueError(
            f"normalize_crop expected a (H, W, 3) array (sagittal replicated to 3 "
            f"channels, axial as 3 real neighboring slices -- see crop_extraction.py), "
            f"got shape {crop_array.shape}. This means a crop was saved in a shape "
            f"that no longer matches the (H, W, 3) contract crop_dataset.py and "
            f"assemble_study_output.py both assume -- check the extraction step that "
            f"produced this array before trusting anything downstream of it."
        )

    img = crop_array.astype(np.float32)
    img = (img - mean) / (std + 1e-8)
    tensor = torch.from_numpy(img).permute(2, 0, 1)
    return tensor.contiguous()

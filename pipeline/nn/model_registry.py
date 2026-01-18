from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ModelLoadResult:
    available: bool
    reason: Optional[str]
    predictor: Optional["ModelPredictor"]
    model_family: Optional[str]
    model_id: Optional[str]
    device: Optional[str]


class ModelPredictor:
    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _resolve_drivable_ids(id2label: Dict[int, str], names: List[str]) -> List[int]:
    if not id2label:
        return []
    name_map = {str(v).strip().lower(): int(k) for k, v in id2label.items()}
    ids = []
    for name in names:
        key = str(name).strip().lower()
        if key in name_map:
            ids.append(name_map[key])
    return sorted(set(ids))


class _TransformersSegPredictor(ModelPredictor):
    def __init__(self, family: str, processor: Any, model: Any, device: str, drivable_ids: List[int]):
        self.family = family
        self.processor = processor
        self.model = model
        self.device = device
        self.drivable_ids = drivable_ids

    def _postprocess_seg(self, outputs: Any, size_hw: tuple[int, int]) -> np.ndarray:
        if self.family == "mask2former":
            seg = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[size_hw]
            )[0]
            return seg.detach().cpu().numpy()
        logits = outputs.logits
        import torch

        logits = torch.nn.functional.interpolate(
            logits, size=size_hw, mode="bilinear", align_corners=False
        )
        return torch.argmax(logits, dim=1)[0].cpu().numpy()

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        import torch

        inputs = self.processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        seg = self._postprocess_seg(outputs, image_rgb.shape[:2])
        if not self.drivable_ids:
            return np.zeros(seg.shape, dtype=bool)
        return np.isin(seg, self.drivable_ids)


def load_model(cfg: Dict[str, Any]) -> ModelLoadResult:
    family = str(cfg.get("model_family") or "").strip().lower()
    model_id = str(cfg.get("model_id") or "").strip()
    if cfg.get("implemented") is False or cfg.get("status") == "not_implemented":
        return ModelLoadResult(False, "not_implemented", None, family, model_id, None)
    if family not in {"segformer", "mask2former"}:
        return ModelLoadResult(False, f"unsupported_family:{family}", None, family, model_id, None)

    try:
        import torch
        from transformers import AutoImageProcessor
        if family == "segformer":
            from transformers import SegformerForSemanticSegmentation as ModelCls
        else:
            from transformers import Mask2FormerForUniversalSegmentation as ModelCls
    except Exception as exc:  # pragma: no cover - dependency gate
        return ModelLoadResult(
            False,
            f"missing_deps:{exc}. Install requirements_nn.txt",
            None,
            family,
            model_id,
            None,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = ModelCls.from_pretrained(model_id).to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", {}) or {}
    drivable_names = cfg.get("postprocess_params", {}).get("drivable_classes") or ["road", "sidewalk"]
    drivable_ids = _resolve_drivable_ids(id2label, list(drivable_names))
    if not drivable_ids:
        return ModelLoadResult(
            False,
            f"no_drivable_classes:{drivable_names}",
            None,
            family,
            model_id,
            device,
        )
    predictor = _TransformersSegPredictor(family, processor, model, device, drivable_ids)
    return ModelLoadResult(True, None, predictor, family, model_id, device)

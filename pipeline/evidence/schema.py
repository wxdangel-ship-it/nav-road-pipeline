from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

PRIMITIVE_REQUIRED_FIELDS = [
    "kind",
    "provider_id",
    "model_id",
    "model_version",
    "source",
    "drive_id",
    "frame_id",
    "geom_crs",
    "geometry",
    "quality",
    "support",
    "uncertainty",
    "provenance",
]

WORLD_CANDIDATE_REQUIRED_FIELDS = [
    "candidate_id",
    "drive_id",
    "frame_id",
    "source",
    "provider_id",
    "version",
    "geom_crs",
    "geometry",
    "quality",
    "support",
    "uncertainty",
    "provenance",
    "rect_w",
    "rect_l",
    "rectangularity",
    "drift_flag",
    "prop_reason",
    "reject_reasons",
]


@dataclass
class Provenance:
    config_path: str = ""
    git_commit: str = ""
    generated_at: str = ""

    @staticmethod
    def now() -> "Provenance":
        return Provenance(generated_at=datetime.now().isoformat(timespec="seconds"))


@dataclass
class SupportStats:
    points_support: int = 0
    points_support_accum: int = 0
    reproj_iou_bbox: Optional[float] = None
    reproj_iou_dilated: Optional[float] = None


@dataclass
class QualityStats:
    score: Optional[float] = None


@dataclass
class UncertaintyStats:
    drift_flag: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrimitiveEvidence:
    kind: str
    provider_id: str
    model_id: str
    model_version: str
    source: str
    drive_id: str
    frame_id: str
    geom_crs: str
    geometry: Any
    quality: QualityStats = field(default_factory=QualityStats)
    support: SupportStats = field(default_factory=SupportStats)
    uncertainty: UncertaintyStats = field(default_factory=UncertaintyStats)
    provenance: Provenance = field(default_factory=Provenance.now)

    backend_status: str = ""
    fallback_used: bool = False
    fallback_from: str = ""
    fallback_to: str = ""
    backend_reason: str = ""
    timestamp: Optional[str] = None


@dataclass
class WorldCandidate:
    candidate_id: str
    drive_id: str
    frame_id: str
    source: str
    provider_id: str
    version: str
    geom_crs: str
    geometry: Any
    quality: QualityStats = field(default_factory=QualityStats)
    support: SupportStats = field(default_factory=SupportStats)
    uncertainty: UncertaintyStats = field(default_factory=UncertaintyStats)
    provenance: Provenance = field(default_factory=Provenance.now)

    rect_w: float = 0.0
    rect_l: float = 0.0
    rectangularity: float = 0.0
    drift_flag: bool = False
    prop_reason: str = ""
    reject_reasons: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None


def validate_required_fields(record: Dict[str, Any], required: List[str]) -> List[str]:
    missing = []
    for key in required:
        if key not in record:
            missing.append(key)
    return missing


def example_primitive_record() -> Dict[str, Any]:
    return {
        "kind": "det",
        "provider_id": "gdino_sam2_v1",
        "model_id": "gdino_sam2_v1",
        "model_version": "v1",
        "source": "image",
        "drive_id": "2013_05_28_drive_0010_sync",
        "frame_id": "000280",
        "geom_crs": "pixel",
        "geometry": {"bbox": [100, 200, 180, 260]},
        "quality": {"score": 0.82},
        "support": {
            "points_support": 5,
            "points_support_accum": 12,
            "reproj_iou_bbox": 0.23,
            "reproj_iou_dilated": 0.31,
        },
        "uncertainty": {"drift_flag": False, "details": {}},
        "provenance": {
            "config_path": "configs/image_model_zoo.yaml",
            "git_commit": "<hash>",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "backend_status": "real",
        "fallback_used": False,
        "fallback_from": "",
        "fallback_to": "",
        "backend_reason": "",
        "timestamp": "",
    }


def example_candidate_record() -> Dict[str, Any]:
    return {
        "candidate_id": "cw_0010_000280_00",
        "drive_id": "2013_05_28_drive_0010_sync",
        "frame_id": "000280",
        "source": "fusion",
        "provider_id": "stage2_video",
        "version": "v1",
        "geom_crs": "utm32",
        "geometry": "POLYGON((...))",
        "quality": {"score": 0.76},
        "support": {
            "points_support": 28,
            "points_support_accum": 64,
            "reproj_iou_bbox": 0.18,
            "reproj_iou_dilated": 0.27,
        },
        "uncertainty": {"drift_flag": False, "details": {"prop_reason": "seed_track_ok"}},
        "provenance": {
            "config_path": "configs/crosswalk_stage2_sam2video.yaml",
            "git_commit": "<hash>",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "rect_w": 2.4,
        "rect_l": 8.6,
        "rectangularity": 0.52,
        "drift_flag": False,
        "prop_reason": "seed_track_ok",
        "reject_reasons": [],
        "timestamp": "",
    }

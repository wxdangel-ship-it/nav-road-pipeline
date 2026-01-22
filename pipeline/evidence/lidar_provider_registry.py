from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Type


_PROVIDERS: Dict[str, Type] = {}


@dataclass
class LidarProviderContext:
    model_cfg: dict
    device: str
    data_root: str


def register_lidar_provider(family: str):
    def _wrap(cls):
        _PROVIDERS[family] = cls
        return cls

    return _wrap


def get_lidar_provider(family: str):
    return _PROVIDERS.get(family)

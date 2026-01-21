from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class ProviderContext:
    model_cfg: dict
    seg_schema: dict
    device: str


class ImageProvider:
    def __init__(self, ctx: ProviderContext) -> None:
        self.ctx = ctx

    def load(self) -> None:
        return None

    def infer(self, images, out_dir):
        raise NotImplementedError

    def dump(self, out_dir) -> None:
        return None


_REGISTRY: Dict[str, Callable[[ProviderContext], ImageProvider]] = {}


def register_provider(family: str):
    def _wrap(cls):
        _REGISTRY[str(family).lower()] = cls
        return cls

    return _wrap


def get_provider(family: str) -> Optional[Callable[[ProviderContext], ImageProvider]]:
    return _REGISTRY.get(str(family).lower())

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type


@dataclass
class ProviderContext:
    config: dict
    index: dict
    device: str = "cpu"


class BaseProvider:
    backend_status: str = "unknown"
    fallback_used: bool = False
    fallback_from: str = ""
    fallback_to: str = ""
    backend_reason: str = ""

    def __init__(self, ctx: ProviderContext) -> None:
        self.ctx = ctx

    def run(self, config: dict, index: dict) -> dict:
        raise NotImplementedError

    def _enforce_strict_backend(self) -> None:
        strict = str(os.environ.get("STRICT_BACKEND", "0")).strip()
        if strict == "1" and self.backend_status != "real":
            raise RuntimeError(f"STRICT_BACKEND=1 but backend_status={self.backend_status}")

    def _backend_report(self) -> dict:
        return {
            "backend_status": self.backend_status,
            "fallback_used": self.fallback_used,
            "fallback_from": self.fallback_from,
            "fallback_to": self.fallback_to,
            "backend_reason": self.backend_reason,
        }


_REGISTRY: Dict[str, Dict[str, Type[BaseProvider]]] = {
    "image": {},
    "lidar": {},
    "sat": {},
    "traj": {},
}


def register_provider(source: str, provider_id: str) -> Callable[[Type[BaseProvider]], Type[BaseProvider]]:
    def _wrap(cls: Type[BaseProvider]) -> Type[BaseProvider]:
        src = str(source).lower()
        _REGISTRY.setdefault(src, {})
        _REGISTRY[src][str(provider_id).lower()] = cls
        return cls

    return _wrap


def get_provider(source: str, provider_id: str) -> Optional[Type[BaseProvider]]:
    return _REGISTRY.get(str(source).lower(), {}).get(str(provider_id).lower())


def list_providers(source: Optional[str] = None) -> Dict[str, Dict[str, Type[BaseProvider]]]:
    if source is None:
        return dict(_REGISTRY)
    return {str(source).lower(): _REGISTRY.get(str(source).lower(), {})}

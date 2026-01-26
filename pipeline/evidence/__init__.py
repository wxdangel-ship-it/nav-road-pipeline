from __future__ import annotations

from pipeline.evidence.schema import (
    PrimitiveEvidence,
    WorldCandidate,
    PRIMITIVE_REQUIRED_FIELDS,
    WORLD_CANDIDATE_REQUIRED_FIELDS,
    example_candidate_record,
    example_primitive_record,
    validate_required_fields,
)
from pipeline.evidence.registry import BaseProvider, ProviderContext, get_provider, list_providers, register_provider

__all__ = [
    "PrimitiveEvidence",
    "WorldCandidate",
    "PRIMITIVE_REQUIRED_FIELDS",
    "WORLD_CANDIDATE_REQUIRED_FIELDS",
    "example_candidate_record",
    "example_primitive_record",
    "validate_required_fields",
    "BaseProvider",
    "ProviderContext",
    "get_provider",
    "list_providers",
    "register_provider",
]

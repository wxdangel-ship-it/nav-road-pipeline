# Architecture Overview

This document summarizes the current architecture and data flow in one page.

## End-to-end Flow

```
Data Sources
  Image / LiDAR / SAT / Traj
        |
        v
Provider Adapters (pluggable)
  -> Primitive Evidence (pixel/world)
        |
        v
Projection + Alignment (fullpose + QA diagnostics)
        |
        v
Fusion / Resolve (M0-M7)
        |
        v
World Candidates (UTM32/WGS84)
```

## Evidence Priority (Arm0-3)
- Arm0: base (LiDAR + video + trajectories)
- Arm1: base + OSM
- Arm2: base + SAT
- Arm3: base + OSM + SAT

Priority rule: strong evidence (LiDAR + in-vehicle video) > trajectories > soft prior (OSM/SAT).
Priors may only "nudge" results and must record conflict_rate when disabled.

## Provider Contract
Providers must emit Primitive Evidence that matches `docs/evidence_schema.md`.
Key requirements:
- Stable `provider_id`, `model_version`, `source`.
- `geom_crs` must be correct (`pixel`, `utm32`, `wgs84`).
- `_wgs84` outputs must be true EPSG:4326.

## Schema Contract
- Primitive Evidence and World Candidates share a stable set of fields.
- Schema lives in `docs/evidence_schema.md` and is the single source of truth.

## Stage2 (SAM2 video) Concept Flow
```
seed -> propagate -> verify -> final
```
- seed: select high-confidence frame candidates.
- propagate: track masks across temporal window.
- verify: enforce IoU/support/rectangularity/ROI constraints.
- final: write world candidates + QA traces (drift_flag/prop_reason).

# CHANGELOG_PHASE (Docs Upgrade)

## Files Updated / Added
- `README.md`: new project goals, data sources/modules, quick start, QA workflow, outputs, FAQ.
- `docs/ARCHITECTURE.md`: architecture overview + evidence priority + Stage2 flow.
- `docs/evidence_schema.md`: dual-layer schema (Primitive Evidence + World Candidates).
- `docs/RUNBOOK.md`: reproducible command set + diagnostics workflow.
- `SPEC.md`: phase milestones and submission boundaries.

## Key New Sections
- Project goals and three-stage outputs
- Provider + schema contract references
- QA workflow (raw/gated/entities) and QA index usage
- Stage2 SAM2 video concept flow
- Minimal command set for strict/quick regressions and diagnostics
- Submission boundary checklist

## Recent Verification
- Date: 2026-01-26
- Branch/Commit: main @ c5f5745
- smoke: OK, run_dir=runs/smoke_20260126_154255
- eval: OK, run_dir=runs/eval_20260126_154259
- Notes: standard smoke + eval --max-frames 2000

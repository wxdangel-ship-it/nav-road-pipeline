# Precommit Checklist

- generated_at: 2026-01-26 13:27:46

## Git status (porcelain)
-  M README.md
-  M SPEC.md
-  M configs/crosswalk_fix_range.yaml
-  M docs/evidence_schema.md
-  M pipeline/evidence/__init__.py
-  M scripts/run_crosswalk_full.cmd
-  M tools/debug_projection_chain.py
-  M tools/run_crosswalk_drive_full.py
-  M tools/run_crosswalk_monitor_range.py
- ?? configs/crosswalk_golden8_full.yaml
- ?? docs/ARCHITECTURE.md
- ?? docs/CHANGELOG_PHASE.md
- ?? docs/RUNBOOK.md
- ?? pipeline/evidence/registry.py
- ?? pipeline/evidence/schema.py
- ?? pipeline/fusion/
- ?? pipeline/projection/
- ?? scripts/precommit_check.cmd
- ?? scripts/run_crosswalk_golden8_full.cmd
- ?? scripts/run_near_final_diagnose.cmd
- ?? scripts/run_validation_250_500.cmd
- ?? tools/_tmp_run_crosswalk_monitor_range.py
- ?? tools/near_final_diagnose.py
- ?? tools/precommit_check.py
- ?? tools/print_schema_example.py
- ?? tools/projection_smoke_250_500.cmd
- ?? tools/run_crosswalk_golden8_full.py
- ?? tools/run_validation_250_500.py

## Staged files (index)
- none

## Forbidden in index (runs/cache/data/weights)
- none

## README/SPEC script path check
- ok

## Manual reminders
- do not add runs/, cache/, data/, or weights to git
- confirm all .cmd paths referenced in README.md/SPEC.md exist
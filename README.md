# nav-road-pipeline
Windows-native road production pipeline (POC->Prod)

## GEOM NN backend (optional)
- Optional deps: `pip install -r requirements_nn.txt`
- Run with NN backend: `set GEOM_BACKEND=nn` then `scripts\\build_geom.cmd ...`
- Default behavior: `GEOM_BACKEND=auto` prefers NN and falls back to algo on missing deps/errors.

## Strong road polygon smoothing
- Enable strong profile in postopt sweep (quick):
  - `scripts\\sweep_geom_postopt.cmd --config configs\\geom_postopt_strong_smooth.yaml --quick-only --quick-max-frames 400`
- Enable strong profile in postopt sweep (full):
  - `scripts\\sweep_geom_postopt.cmd --config configs\\geom_postopt_strong_smooth.yaml --full-only --full-max-frames 2000`
- Direct single run (ad hoc):
  - `set SMOOTH_PROFILE=strong` then `scripts\\build_geom.cmd --drive <drive_id> --max-frames 400`

## Smoothness metrics
- `roughness` (perimeter^2 / area) and `vertex_count` are logged in `GeomSummary.json`/`qc.json`.
- Lower `roughness` and lower `vertex_count` imply smoother, more regularized boundaries.

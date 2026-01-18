# nav-road-pipeline
Windows-native road production pipeline (POC->Prod)

## GEOM NN backend (optional)
- Optional deps: `pip install -r requirements_nn.txt`
- Run with NN backend: `set GEOM_BACKEND=nn` then `scripts\\build_geom.cmd ...`
- Default behavior: `GEOM_BACKEND=auto` prefers NN and falls back to algo on missing deps/errors.

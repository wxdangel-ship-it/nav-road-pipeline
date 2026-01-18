from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import rasterio
except Exception:
    rasterio = None


def _scan_tiles(tiles_dir: Path) -> List[Dict[str, float]]:
    if rasterio is None:
        raise RuntimeError("missing_rasterio")
    items: List[Dict[str, float]] = []
    for ext in ("*.tif", "*.tiff", "*.jp2", "*.jpg", "*.jpeg"):
        for path in tiles_dir.rglob(ext):
            try:
                with rasterio.open(path) as ds:
                    bounds = ds.bounds
                items.append(
                    {
                        "path": str(path),
                        "minx": float(bounds.left),
                        "miny": float(bounds.bottom),
                        "maxx": float(bounds.right),
                        "maxy": float(bounds.top),
                    }
                )
            except Exception:
                continue
    return items


def _read_xlsx(path: Path) -> Optional[List[Dict[str, float]]]:
    if pd is None:
        return None
    try:
        df = pd.read_excel(path)
    except Exception:
        return None
    cols = {c.lower(): c for c in df.columns}
    file_col = None
    for key in ("path", "file", "filename", "tile", "name"):
        if key in cols:
            file_col = cols[key]
            break
    bounds_keys = {
        "minx": cols.get("minx") or cols.get("xmin") or cols.get("left"),
        "miny": cols.get("miny") or cols.get("ymin") or cols.get("bottom"),
        "maxx": cols.get("maxx") or cols.get("xmax") or cols.get("right"),
        "maxy": cols.get("maxy") or cols.get("ymax") or cols.get("top"),
    }
    if not file_col or not all(bounds_keys.values()):
        return None
    items: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        try:
            items.append(
                {
                    "path": str(row[file_col]),
                    "minx": float(row[bounds_keys["minx"]]),
                    "miny": float(row[bounds_keys["miny"]]),
                    "maxx": float(row[bounds_keys["maxx"]]),
                    "maxy": float(row[bounds_keys["maxy"]]),
                }
            )
        except Exception:
            continue
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Build DOP20 tile index cache.")
    ap.add_argument("--dop20-root", default=r"E:\KITTI360\KITTI-360\_lglbw_dop20")
    ap.add_argument("--tiles-dir", default="tiles_utm32")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dop20_root = Path(args.dop20_root)
    tiles_dir = dop20_root / args.tiles_dir
    if not tiles_dir.exists():
        raise SystemExit(f"ERROR: tiles dir not found: {tiles_dir}")

    items = None
    xlsx_path = dop20_root / "tiles_index.xlsx"
    if xlsx_path.exists():
        items = _read_xlsx(xlsx_path)

    if items is None:
        items = _scan_tiles(tiles_dir)

    out_path = Path(args.out) if args.out else dop20_root / "dop20_tiles_index.json"
    payload = json.dumps(items, ensure_ascii=False, indent=2)
    try:
        out_path.write_text(payload, encoding="utf-8")
        print(f"[DOP20] wrote {out_path} ({len(items)} tiles)")
    except PermissionError:
        fallback = Path("cache") / "dop20_tiles_index.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(payload, encoding="utf-8")
        print(f"[DOP20] wrote {fallback} ({len(items)} tiles) [fallback]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

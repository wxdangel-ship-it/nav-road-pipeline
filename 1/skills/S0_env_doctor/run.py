import json
import sys
from pathlib import Path

from core.debug_layers import write_demo_debug_gpkg
from core.reporting import write_metrics, write_skill_report


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return "unknown"


def main(run_dir: Path):
    """环境自检 Skill：生成最小闭环四件套（除 RunMeta 由上层 runner 写入）。"""
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "python": sys.version.replace("\n", " "),
        "packages": {
            "numpy": _pkg_version("numpy"),
            "shapely": _pkg_version("shapely"),
            "geopandas": _pkg_version("geopandas"),
            "pyogrio": _pkg_version("pyogrio"),
            "pyproj": _pkg_version("pyproj"),
            "rasterio": _pkg_version("rasterio"),
            "laspy": _pkg_version("laspy"),
        },
        "checks": {},
        "failure_bucket": None,
    }

    # 1) import checks
    try:
        import geopandas  # noqa: F401
        metrics["checks"]["geopandas_import_ok"] = True
    except Exception as e:
        metrics["checks"]["geopandas_import_ok"] = False
        metrics["failure_bucket"] = "import_error"
        write_metrics(run_dir, metrics)
        write_skill_report(run_dir, [
            "# S0_env_doctor Report",
            "",
            "## 失败：geopandas 导入失败",
            f"- error: `{type(e).__name__}: {e}`",
        ])
        raise

    try:
        import rasterio  # noqa: F401
        metrics["checks"]["rasterio_import_ok"] = True
    except Exception:
        metrics["checks"]["rasterio_import_ok"] = False

    try:
        import laspy  # noqa: F401
        metrics["checks"]["laspy_import_ok"] = True
    except Exception:
        metrics["checks"]["laspy_import_ok"] = False

    # 2) write gpkg
    out_gpkg = run_dir / "debug_layers.gpkg"
    try:
        write_demo_debug_gpkg(out_gpkg, epsg=3857, bbox=(0, 0, 100, 100))
        metrics["checks"]["gpkg_write_ok"] = True
    except Exception as e:
        metrics["checks"]["gpkg_write_ok"] = False
        metrics["failure_bucket"] = "gpkg_write_error"
        write_metrics(run_dir, metrics)
        write_skill_report(run_dir, [
            "# S0_env_doctor Report",
            "",
            "## 失败：debug_layers.gpkg 写入失败",
            f"- error: `{type(e).__name__}: {e}`",
            "",
            "建议：优先检查 `pyogrio` 是否安装成功；必要时重建虚拟环境后重装 requirements。",
        ])
        raise

    # 3) finalize metrics + report
    write_metrics(run_dir, metrics)
    write_skill_report(run_dir, [
        "# S0_env_doctor Report",
        "",
        "## 结论",
        "- 环境自检通过：Geo 库可导入，debug gpkg 可写入。",
        "",
        "## 关键版本",
        "```json",
        json.dumps(metrics["packages"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 产物",
        f"- debug_layers.gpkg: `{out_gpkg.name}`（建议用 QGIS 打开验证）",
    ])

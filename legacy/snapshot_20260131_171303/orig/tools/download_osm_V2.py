# -*- coding: utf-8 -*-
"""
功能：
    根据给定 WGS84 BBOX（经纬度范围）从 Overpass API 下载 OSM 车行道路（drivable roads），并输出为 GeoJSON。
    仅保留“可供机动车通行”的 highway 类型（可在参数区调整），并尽量过滤 access/motor_vehicle/vehicle=no 的道路。

输入：
    - BBOX：minlon/minlat/maxlon/maxlat（WGS84，经纬度）
    - Overpass API（联网请求）

输出：
    - drivable_roads.geojson：车行道路 LineString FeatureCollection（EPSG:4326）
    - overpass_raw.json：Overpass 原始 JSON（便于复核/重解析）

依赖：
    - requests（推荐：pip install requests）

参数：
    - 见下方“参数区”，脚本可直接在 PyCharm 运行，无需命令行参数
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple, Optional

try:
    import requests
except Exception as e:
    requests = None


# =============================================================================
# 参数区（按需修改）
# =============================================================================

# 1) 下载范围（WGS84）
BBOX = {
    "minlon": 8.375340151608908,
    "minlat": 48.941449366025495,
    "maxlon": 8.498286473216496,
    "maxlat": 49.03020089622681,
}

# 2) 输出目录（Windows 路径请用原始字符串）
OUT_DIR = r"E:\KITTI360\KITTI-360\_osm_download"
OUT_GEOJSON = os.path.join(OUT_DIR, "drivable_roads.geojson")
OUT_RAW_JSON = os.path.join(OUT_DIR, "overpass_raw.json")

OVERWRITE = True

# 3) Overpass 端点（有时会拥堵，失败可切换）
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
OVERPASS_TIMEOUT_SEC = 300
RETRY_TIMES = 5
RETRY_BACKOFF_SEC = 8

# 4) 车行道路 highway 类型（可按需增减）
# 说明：track 是否算车行道路看业务口径，这里默认不包含 track。
HIGHWAY_ALLOW_LIST = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
]

INCLUDE_TRACK = False  # True 则把 track 加入 allow list
EXCLUDE_SERVICE_PARKING_AISLE = True  # service=parking_aisle 通常是停车场车道，可按需排除

# 5) access 过滤（尽量过滤禁行/私有）
# 注意：OSM 标签比较复杂，本脚本做“尽量过滤”，不保证 100% 精确。
DENY_REGEX = "no|private"

# =============================================================================
# 函数区
# =============================================================================

def setup_logger() -> logging.Logger:
    """初始化日志。"""
    logger = logging.getLogger("osm_drivable_download")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def ensure_out_dir(out_dir: str) -> None:
    """确保输出目录存在。"""
    os.makedirs(out_dir, exist_ok=True)


def safe_remove(path: str, logger: logging.Logger) -> None:
    """如文件存在则删除（受 OVERWRITE 控制）。"""
    if os.path.exists(path):
        os.remove(path)
        logger.info("已删除旧文件：%s", path)


def build_highway_regex(allow_list: List[str]) -> str:
    """将 highway allow_list 转为 overpass 正则。"""
    # 防止 list 里包含正则特殊字符，这里只做简单 join
    return "|".join(sorted(set(allow_list)))


def build_overpass_query(bbox: Dict[str, float], logger: logging.Logger) -> str:
    """构造 Overpass QL 查询：仅车行道路 ways + 关联 nodes。"""
    allow_list = list(HIGHWAY_ALLOW_LIST)
    if INCLUDE_TRACK and "track" not in allow_list:
        allow_list.append("track")

    highway_re = build_highway_regex(allow_list)

    # bbox 顺序：south, west, north, east
    s = bbox["minlat"]
    w = bbox["minlon"]
    n = bbox["maxlat"]
    e = bbox["maxlon"]

    # 附加过滤：尽量过滤 access/vehicle/motor_vehicle 禁行
    # 注意：Overpass 里 [key!~"..."] 表示 key 不匹配该正则（key 不存在也算通过）
    extra_filters = [
        f'["access"!~"{DENY_REGEX}"]',
        f'["vehicle"!~"{DENY_REGEX}"]',
        f'["motor_vehicle"!~"{DENY_REGEX}"]',
    ]

    if EXCLUDE_SERVICE_PARKING_AISLE:
        # service=parking_aisle 过滤（同理：不存在该 tag 也通过）
        extra_filters.append('["service"!="parking_aisle"]')

    filt = "".join(extra_filters)

    q = f"""
[out:json][timeout:{OVERPASS_TIMEOUT_SEC}];
(
  way["highway"~"^({highway_re})$"]{filt}({s},{w},{n},{e});
);
out body;
>;
out skel qt;
""".strip()

    logger.info("Overpass 查询 highway 类型：%s", highway_re)
    return q


def post_overpass(query: str, endpoints: List[str], logger: logging.Logger) -> Dict[str, Any]:
    """向 Overpass 发送请求（带重试与端点切换）。"""
    if requests is None:
        raise RuntimeError("缺少 requests 依赖。请先执行：pip install requests")

    headers = {
        "User-Agent": "nav-road-pipeline-osm-downloader/1.0 (contact: local)",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }

    last_err: Optional[Exception] = None
    for ep in endpoints:
        logger.info("尝试 Overpass 端点：%s", ep)
        for i in range(RETRY_TIMES):
            try:
                resp = requests.post(ep, data={"data": query}, headers=headers, timeout=OVERPASS_TIMEOUT_SEC)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                if not isinstance(data, dict) or "elements" not in data:
                    raise RuntimeError("Overpass 返回格式异常（缺少 elements）")
                return data
            except Exception as e:
                last_err = e
                wait = RETRY_BACKOFF_SEC * (i + 1)
                logger.warning("Overpass 请求失败（端点=%s，第%d/%d次）：%s；%ds 后重试",
                               ep, i + 1, RETRY_TIMES, str(e), wait)
                time.sleep(wait)

        logger.warning("端点 %s 多次失败，切换下一个端点。", ep)

    raise RuntimeError(f"所有 Overpass 端点均失败，最后错误：{last_err}")


def is_way_drivable(tags: Dict[str, Any]) -> bool:
    """二次过滤：判断 way 是否为车行道路（尽量过滤禁行）。"""
    if not tags:
        return False

    highway = str(tags.get("highway", "")).strip()
    allow = set(HIGHWAY_ALLOW_LIST)
    if INCLUDE_TRACK:
        allow.add("track")
    if highway not in allow:
        return False

    # 禁行过滤（尽量）
    for k in ("access", "vehicle", "motor_vehicle"):
        v = str(tags.get(k, "")).strip().lower()
        if v in ("no", "private"):
            return False

    if EXCLUDE_SERVICE_PARKING_AISLE:
        if highway == "service" and str(tags.get("service", "")).strip().lower() == "parking_aisle":
            return False

    return True


def overpass_json_to_geojson(overpass: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """将 Overpass JSON 转为 GeoJSON FeatureCollection（LineString）。"""
    elements = overpass.get("elements", [])
    node_map: Dict[int, Tuple[float, float]] = {}

    # 先收集 nodes
    for el in elements:
        if el.get("type") == "node":
            nid = el.get("id")
            lat = el.get("lat")
            lon = el.get("lon")
            if isinstance(nid, int) and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                node_map[nid] = (float(lon), float(lat))

    features: List[Dict[str, Any]] = []
    total_ways = 0
    kept_ways = 0
    skipped_missing_geom = 0

    for el in elements:
        if el.get("type") != "way":
            continue

        total_ways += 1
        tags = el.get("tags", {}) or {}
        if not is_way_drivable(tags):
            continue

        nodes = el.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) < 2:
            skipped_missing_geom += 1
            continue

        coords: List[List[float]] = []
        miss = 0
        for nid in nodes:
            pt = node_map.get(nid)
            if pt is None:
                miss += 1
                continue
            coords.append([pt[0], pt[1]])

        if len(coords) < 2:
            skipped_missing_geom += 1
            continue

        kept_ways += 1

        props = dict(tags)
        props.update({
            "osm_id": el.get("id"),
            "osm_type": "way",
            "node_count": len(nodes),
            "node_missing": miss,
        })

        feat = {
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            }
        }
        features.append(feat)

    logger.info("Overpass ways 总数=%d，保留车行ways=%d，几何不足/缺node=%d",
                total_ways, kept_ways, skipped_missing_geom)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def save_json(obj: Any, path: str, logger: logging.Logger) -> None:
    """保存 JSON/GeoJSON。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    logger.info("已写出：%s", path)


def main() -> None:
    logger = setup_logger()
    ensure_out_dir(OUT_DIR)

    if OVERWRITE:
        for p in (OUT_GEOJSON, OUT_RAW_JSON):
            safe_remove(p, logger)
    else:
        for p in (OUT_GEOJSON, OUT_RAW_JSON):
            if os.path.exists(p):
                raise FileExistsError(f"输出已存在且 OVERWRITE=False：{p}")

    logger.info("BBOX(WGS84)：minlon=%.12f minlat=%.12f maxlon=%.12f maxlat=%.12f",
                BBOX["minlon"], BBOX["minlat"], BBOX["maxlon"], BBOX["maxlat"])

    query = build_overpass_query(BBOX, logger)
    overpass = post_overpass(query, OVERPASS_ENDPOINTS, logger)

    save_json(overpass, OUT_RAW_JSON, logger)

    geojson = overpass_json_to_geojson(overpass, logger)
    save_json(geojson, OUT_GEOJSON, logger)

    logger.info("完成：车行道路 GeoJSON 已生成。")


if __name__ == "__main__":
    main()

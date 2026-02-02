# Skill: surface_evidence_utm32 (v0.1.0)

## 目标
基于 Skill#1 的融合点云产物（UTM32/EPSG:32632），输出地表证据产品：路面点云、DEM/质量栅格、路面矢量、BEV 标线特征。
本 Skill 仅消费融合产物，不触发 fusion。

## 输入来源（Fusion Source）
支持三种方式：
1) baseline_active：读取 `baselines/ACTIVE_LIDAR_FUSION_BASELINE.txt` 指向目录，解析 manifest 得到 LAZ 路径与 bbox/transform。
2) run_dir：用户显式提供某次 fusion_run_dir，读取 `outputs/fused_points_utm32_part_*.laz`（或单文件）。
3) explicit：用户提供 LAZ 文件路径（或 glob），并可选提供 bbox/transform。

## 入口
`python -m scripts.run_skill_surface_evidence_utm32`

配置方式：
- job YAML：`configs/jobs/surface_evidence/*.yaml`
- 固定参数：`configs/skills/surface_evidence_utm32.yaml`

## 输出
run_dir 结构（小文件证据 + 大文件清单）：
- outputs/
  - road_surface_points_utm32.laz
  - road_surface_points_utm32.meta.json
  - surface_dem_utm32.tif
  - surface_dem_quality_utm32.tif
  - surface_dem_preview.png
  - road_surface_polygon_utm32.gpkg
  - road_surface_polygon_preview.geojson
  - bev_markings_utm32_tiles_r005m/ (tile tif + preview)
  - bev_markings_tiles_index_r005m.geojson
  - bev_rois_r005m/ (可选)
  - large_files_manifest.json
- report/
  - metrics.json
  - gates.json
  - params.json
- logs/
  - run.log / run_tail.log

## 门禁（Gates）
- epsg==32632 且 bbox_check==ok
- points_road_surface > 0 且 ratio_road_surface >= 0.02
- DEM 输出时：dem_valid_ratio >= 0.30
- BEV 输出时：tiles_count > 0 且 empty_tile_ratio < 0.9

## 注意事项
- Skill#2 绝不调用 Skill#1；只消费融合产物。
- 大文件（LAZ/TIF/GPKG）不入库，仅记录 manifest 与 hash_head。

## BEV Tiles 默认参数
- tiles res_m 默认 0.05m，可�?job YAML ���?`bev.tiles.res_m`（例�?0.10/0.20�?
- tile_size_px 默认 2048，可�?job YAML ���?`bev.tiles.tile_size_px`
- overlap_px 默认 0，可�?job YAML ���?`bev.tiles.overlap_px`

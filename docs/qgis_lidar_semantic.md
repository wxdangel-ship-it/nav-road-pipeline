# QGIS: LiDAR 语义点云（road / markings / crosswalk）检查指引

本指引用于快速检查 `runs\lidar_semantic_<run_id>\` 的语义结果。

## 1) 运行入口

```bat
scripts\run_lidar_semantic_golden8.cmd
```

## 2) 推荐加载顺序（单条带）

以某条带 `drives\<drive_id>\` 为例：

1. 矢量（优先）
   - `vectors\road_surface_utm32.gpkg`
   - `vectors\markings_utm32.gpkg`
   - `vectors\crosswalk_utm32.gpkg`
2. 栅格（辅助）
   - `rasters\road_mask_utm32.tif`
   - `rasters\marking_mask_utm32.tif`
   - `rasters\intensity_max_utm32.tif`
3. 点云（可选）
   - `pointcloud\semantic_points_utm32.laz`
   - `pointcloud\road_surface_points_utm32.laz`
   - `pointcloud\markings_points_utm32.laz`

说明：
- 当前点云文件以 LAS 格式写出（同时生成 `.las` 版本），QGIS 中建议优先加载同名 `.las`。
- 所有数据均为 EPSG:32632。

## 3) 推荐样式

### 3.1 road_surface
- 填充：浅灰，透明度 40%
- 轮廓：深灰细线

### 3.2 markings
- 填充：亮黄，透明度 20%
- 轮廓：亮黄细线

### 3.3 crosswalk
- 填充：青色或亮绿，透明度 35%
- 轮廓：同色加粗

## 4) Golden8 总览

直接加载 merged 层：

- `merged\road_surface_utm32.gpkg`
- `merged\markings_utm32.gpkg`
- `merged\crosswalk_utm32.gpkg`
- `merged\qa_index.csv`

建议按 `drive_id` 分类渲染，快速定位异常条带。


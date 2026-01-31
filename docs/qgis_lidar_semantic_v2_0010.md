# QGIS quick check (LiDAR semantic v2 0010)

1) Load point layers:
- `road_surface_points_utm32.laz` (or `.las`)
- `non_road_points_utm32.laz` (or `.las`)
- `markings_points_utm32.laz` (or `.las`)
- `crosswalk_points_utm32.laz` (or `.las`, optional)

2) Load raster layers:
- `marking_score_utm32.tif` (use 2–98% stretch for contrast)
- `marking_mask_utm32.tif`
- `road_mask_utm32.tif`

3) Load vector layers:
- `road_surface_utm32.gpkg`
- `markings_utm32.gpkg`
- `crosswalk_utm32.gpkg`

All outputs are under:
`runs/lidar_semantic_v2_0010_<run_id>/drives/<drive_id>/`

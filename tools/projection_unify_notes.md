# Projection Unify Notes (image crosswalk pipeline)

## 搜索到的旧口径（被替换）

- `tools/run_crosswalk_monitor_range.py`
  - `_project_world_to_image`: 通过 `pipeline.projection.projector.world_geom_to_image` 使用 `k/r_rect/t_velo_to_cam` 直接投影
  - `_pixel_to_world` / `_project_geometry_ground_plane`: 基于 `k + R_rect + t_cam_to_velo` + 旧 pose，假设 `z=0` 平面求交
- `tools/project_feature_store_to_map.py`
  - `_project_velodyne_to_image`: 直接用 `k/r_rect/t_velo_to_cam` 投影
  - `_pixel_to_world` / `_project_geometry_ground_plane`: 同上，使用旧 pose + ground plane
- `tools/run_crosswalk_drive_full.py`
  - `_project_world_to_image`/`_geom_to_image_points`: 使用 `world_geom_to_image` + 旧 calib/pose

## 新口径（黄金链路）

统一改为：
- 标定/pose 读取：`pipeline/calib/io_kitti360_calib.py`
  - `load_kitti360_calib_bundle`（K, R_rect, P_rect, T_C0_V）
  - `load_cam0_pose_provider`（cam0_to_world）
- 投影：`pipeline/calib/kitti360_projection.py`
  - `project_world_to_image`（world -> cam0 -> rect -> pixel）
- 回投影：`pipeline/calib/kitti360_backproject.py`
  - `world_to_pixel_cam0(frame_id, xyz_world)`
  - `pixel_to_world_on_ground(frame_id, u, v, ground_model)`
    - 优先 DTM（`lidar_clean_dtm`），无 DTM 时使用固定平面 `z=Z0`

## 具体替换点

- `tools/run_crosswalk_monitor_range.py`
  - `_project_world_to_image` → `world_to_pixel_cam0`
  - `_project_geometry_ground_plane` → `pixel_to_world_on_ground`
  - 生成 per-frame `frames/<frame>/crosswalk_frame_utm32.gpkg` 与 `merged/crosswalk_candidates_utm32.gpkg`
- `tools/project_feature_store_to_map.py`
  - `_project_velodyne_to_image` → `kitti360_projection.project_velo_to_image`
  - `_project_geometry_ground_plane` → `pixel_to_world_on_ground`
- `tools/run_crosswalk_drive_full.py`
  - `_project_world_to_image`/`_geom_to_image_points` → `world_to_pixel_cam0`

## 注意

- 所有像素↔世界转换均使用同一套 `T_W_C0` + intrinsics/rect，不再混用旧 fullpose / 手写地面平面求交。
- DTM 优先级：若 `dtm_median_utm32.tif` 可用 → 使用；否则 fallback 到固定平面 `z=Z0`。

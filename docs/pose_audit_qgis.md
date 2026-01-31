# Pose Audit QGIS Notes

1) Load `runs/pose_audit_0010_f250_500_<run_id>/gis/pose_audit_utm32.gpkg`.
2) Check layers:
   - `traj_cam0`
   - `traj_velo_from_cam`
   - `traj_velo_from_imu` (if present)
   - `frame_points`
3) Use attribute `delta_t_m` / `delta_rot_deg` on `frame_points` for spot checks.

Notes:
- CRS is EPSG:32632.
- If `traj_velo_from_imu` is missing, the IMU->velo calibration file was not found.

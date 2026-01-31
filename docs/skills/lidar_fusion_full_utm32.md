# Skill: lidar_fusion_full_utm32 (v0.1.0)

## 目标
将 KITTI-360 指定条带指定帧范围的 LiDAR 逐帧融合，输出 UTM32 (EPSG:32632) 的 LAZ（支持分块），用于 QGIS/CloudCompare 人工质检与后续产线。

## 入口
- python -m scripts.run_skill_lidar_fusion_full_utm32

> 参数采用“脚本参数区集中定义”的方式配置（不依赖命令行参数）。

## 使用方式
### 单条带 / 单帧段（single job）
1) 复制一个 job YAML（如 `configs/jobs/lidar_fusion/0010_f000_300.yaml`）
2) 修改 `drive_id`、`frame_start`、`frame_end` 与 `transform_json`
3) 在 Runner 顶部参数区设置：
   - `MODE = "single"`
   - `JOB_FILE = r"configs\\jobs\\lidar_fusion\\你的_job.yaml"`
4) 运行：
   - `python -m scripts.run_skill_lidar_fusion_full_utm32`

### Golden8 批处理（batch jobs）
1) 在 `configs/jobs/lidar_fusion/golden8_full.yaml` 补齐 jobs 列表
2) 在 Runner 顶部参数区设置：
   - `MODE = "batch"`
   - `BATCH_FILE = r"configs\\jobs\\lidar_fusion\\golden8_full.yaml"`
3) 运行：
   - `python -m scripts.run_skill_lidar_fusion_full_utm32`

> 任意条带/帧范围：只需复制 job YAML 并改 `drive_id`/`frame_start`/`frame_end`。

### transform.mode 用法
- auto_fit：在本次 run_dir/report/ 生成 world_to_utm32_report.json 等小文件，按 gate 判定 PASS/WARN/FAIL
- use_file：使用已有 transform 文件（相对 JOB_DIR 或绝对路径）
- none：不做 utm32，强制输出 world

支持路径占位：`%KITTI_ROOT%`、`${REPO_ROOT}`、`${JOB_DIR}`、`${RUN_DIR}` 等。

## 输入
- KITTI_ROOT：KITTI-360 根目录
- DRIVE_ID：例如 2013_05_28_drive_0010_sync
- FRAME_START/FRAME_END：例如 0–3835
- STRIDE：默认 1

## 核心约束（硬规则）
1) 姿态来源：默认使用 data_poses/<drive>/cam0_to_world.txt（rectified cam0->world）
2) Rectified 一致性：必须乘 R_rect_00（来自 calibration/perspective.txt）
   - T_velo_to_world = T_rectcam0_to_world @ T_rect @ T_velo_to_cam0
3) 强度：float -> uint16，规则 *65535，避免导出 intensity 全 0
4) UTM32：EPSG=32632 + bbox_check 必须通过（不得产生“假 utm32”）
5) 条带门禁：utm32 min_nonzero_dy <= 0.01（避免 float32 精度塌陷造成 0.5m 网格）

## 输出
run_dir 下典型结构：
- outputs/
  - fused_points_utm32_part_*.laz（或单文件 fused_points_utm32.laz）
  - fused_points_utm32_index.json（分块索引）
  - fused_points_utm32.meta.json
  - missing_frames.csv / missing_summary.json
  - bbox_utm32.geojson
- report/
  - metrics.json
  - banding_audit*.json
- logs/
  - run.log / run_tail.log

## 质检门禁（必须满足）
- intensity_max > 0 且 intensity_nonzero_ratio > 0.5
- epsg == 32632 且 bbox_check == ok
- banding.min_nonzero_dy <= 0.01（期望 ~0.001）
- transform gate：PASS/WARN（WARN 必须人工质检通过才能冻结产线基线）

## 产线基线（Baseline）
- ACTIVE 指针：baselines/ACTIVE_LIDAR_FUSION_BASELINE.txt
- baseline 仅保存“小证据文件 + LAZ parts manifest(hash_head)”；不入库 LAZ 本体。

## 常见失败原因
- missing_pose_frame：cam0_to_world 缺帧（当前策略不启用 poses 兜底）
- laz 后端缺失：未安装 lazrs（REQUIRE_LAZ=True 会直接失败）
- pyproj 缺失：无法拟合 world->utm32 transform
- KITTI_ROOT 未设置：请设置系统环境变量或在 job 中写绝对路径
- transform file 找不到：检查 use_file 路径与占位
- transform gate FAIL：rms 超过 warn 门限，停止融合

# LiDAR Fusion Jobs (UTM32)

本目录用于以 YAML 驱动 `lidar_fusion_full_utm32` Skill。新增条带/帧范围时复制一个 job YAML 并修改字段即可。

## Job 字段说明
- kitti_root: KITTI-360 根目录
- drive_id: 例如 2013_05_28_drive_0010_sync
- frame_start / frame_end: 帧范围（闭区间）
- stride: 步长
- output_mode: "utm32" | "world" | "auto"
- output_format: "laz" | "las" | "npz"
- require_laz: true 表示写不出 LAZ 直接失败
- use_r_rect_with_cam0_to_world: cam0_to_world 分支是否乘 R_rect_00
- require_cam0_to_world: 缺 cam0_to_world 是否直接失败
- allow_poses_fallback: 是否允许 poses.txt 兜底
- enable_chunking: 是否分块输出
- max_parts / target_laz_mb_per_part: 分块控制
- banding_max_min_nonzero_dy_m: 条带门禁（建议 0.01）

### transform 段（三态）
```
transform:
  mode: auto_fit        # auto_fit | use_file | none
  file: ""              # mode=use_file 时填写；支持相对 JOB_DIR 或绝对路径；支持占位
  sample_max_frames: 300
  gate_pass_m: 1.0
  gate_warn_m: 1.5
```

## 路径占位与解析规则
- 支持占位：
  - %VAR%（Windows env）
  - ${VAR}（同 env）
  - ${REPO_ROOT}
  - ${JOB_DIR}
  - ${RUN_DIR}
  - ${KITTI_ROOT}
- 相对路径解析：
  - transform.file 若为相对路径：以 JOB_DIR 为基准
  - kitti_root 若为相对路径：以 REPO_ROOT 为基准

推荐写法：
```
kitti_root: "%KITTI_ROOT%"
```
若未设置系统环境变量 KITTI_ROOT，Runner 会报错提示。

## Golden8 批处理
`golden8_full.yaml` 使用 `jobs[*].path` 列表引用多个 job 文件。先复制并补齐 TODO 项即可。

## 依赖说明
Runner 依赖 PyYAML 读取配置；若缺失请安装：

```
python -m pip install pyyaml
```

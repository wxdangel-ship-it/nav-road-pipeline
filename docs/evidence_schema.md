# 统一 Evidence Schema（图像证据层）

本规范用于将不同图像基模型的输出统一到同一套可追溯、可抽检、可对比的证据格式。**下游仅依赖该 schema，不针对每个模型写适配分支**。

## 1. 输出形态与目录

- 每个 provider 产出：
  - 模型原始输出：`runs/<exp_id>/model_outputs/<model_id>/<drive_id>/`
  - 调试可视化：`runs/<exp_id>/debug/<model_id>/<drive_id>/`
  - 统一证据：`runs/<exp_id>/evidence/<model_id>.jsonl`
- `evidence/*.jsonl` 为行式 JSON；每行是一个 record。

## 2. record 字段（最小集合）

- `kind`: 记录类型，`seg_map` 或 `det`
- `provider_id`: provider 名（例如 `gdino_sam2_v1`）
- `model_id`: 模型 ID（与 `provider_id` 一致或更细粒度）
- `model_version`: 模型版本字符串
- `ckpt_hash`: 权重标识（文件名或 hash）
- `backend_status`: `real` / `fallback` / `unavailable`
- `fallback_used`: 是否启用 fallback（bool）
- `fallback_from`: 触发 fallback 的 provider（如 `sam3_v1`）
- `fallback_to`: fallback 实际运行的 provider（如 `gdino_sam2_v1`）
- `backend_reason`: 可选，`missing_dependency` / `weights_not_found` / `runtime_error` 等
- `scene_profile`: `car/aerial/sat` 等
- `drive_id`, `frame_id`
- `image_path`: 原始图像路径（本地）
- `prompt`, `prompt_type`: `text/box/point/auto` 等
- `score`: 置信度（det 记录）
- `bbox`: `[x1,y1,x2,y2]`（像素坐标，det 记录）
- `mask`: 掩码信息（seg_map 记录）
  - `format`: `class_id_png` / `png` / `rle`
  - `path`: 掩码文件路径（本地）
- `geometry_frame`: 当前固定为 `image_px`
- `debug_assets_path`: 叠加图/中间输出路径
- `config_path`: 运行时配置路径（如 `configs/image_model_zoo.yaml`）
- `git_commit`: 当前代码版本（commit hash）
- `generated_at`: 生成时间（ISO）

## 3. 语义与约束

- `kind=seg_map`：`mask.format=class_id_png` 时，PNG 为单通道语义图（像素值为 class_id）。  
- `kind=det`：仅记录 bbox 与 score；若有实例 mask，可补充 `mask` 字段。
- `_wgs84` 命名硬规则：**任何文件名带 `_wgs84` 的输出必须是真 EPSG:4326**。如不是，必须重投影或改名 `_utm32`。

## 4. 追溯字段建议

运行级别建议在 `runs/<exp_id>/run_manifest.json` 中记录：
- `providers` / `sample_index` / `seg_schema` / `feature_schema`
- `device` / `cuda_available`
- `git_commit`

## 5. 示例

```json
{
  "kind": "det",
  "provider_id": "gdino_sam2_v1",
  "model_id": "gdino_sam2_v1",
  "model_version": "v1",
  "ckpt_hash": "sam2.1_hiera_base_plus.pt",
  "scene_profile": "car",
  "drive_id": "2013_05_28_drive_0007_sync",
  "frame_id": "000000",
  "image_path": "D:\\KITTI360\\data_2d_raw\\...\\000000.png",
  "prompt": "crosswalk",
  "prompt_type": "text",
  "score": 0.82,
  "bbox": [100, 200, 180, 260],
  "geometry_frame": "image_px",
  "debug_assets_path": "runs\\...\\debug\\gdino_sam2_v1\\...\\000000_overlay.png",
  "config_path": "configs\\image_model_zoo.yaml",
  "git_commit": "<hash>",
  "generated_at": "2026-01-21T23:45:11"
}
```

# nav-road-pipeline
Windows-native road production pipeline (POC->Prod)

## 快速开始（Windows）

1) 设置数据根目录：
```
set POC_DATA_ROOT=D:\KITTI360
```
2) 一键运行图像证据层（多 provider）：
```
scripts\run_image_providers.cmd
```
3) 生成对比报告（AB）：
```
scripts\run_ab_eval.cmd
```

## 图像证据层入口

- 运行 provider：
  - `scripts/run_image_providers.cmd`
  - 环境变量：`STRICT_BACKEND=1` 禁止 fallback
- 生成 AB 报告：
  - `scripts/run_ab_eval.cmd`
  - 环境变量：`EXCLUDE_FALLBACK_PROVIDERS=1` 默认剔除 fallback
- SAM3 权重准备：
  - `scripts/setup_sam3_weights.cmd`

## Provider 配置

- 配置入口：`configs/image_model_zoo.yaml`
- 重要字段：
  - `backend=auto|real|fallback`
  - `weights_path` / `weights_url`
  - `sam3_builder` / `sam3_predictor` / `sam3_model_cfg`

## Evidence Schema（新增字段）

统一 schema 见 `docs/evidence_schema.md`，新增后端字段：
`backend_status` / `fallback_used` / `fallback_from` / `fallback_to` / `backend_reason`

## 产物目录（典型）

```
runs/<exp_id>/
  model_outputs/
  debug/
  evidence/
  ab_eval/
```

## 约束

- `runs/`、`cache/`、权重与大文件不入库
- `_wgs84` 必须是 EPSG:4326，否则重投影或改名

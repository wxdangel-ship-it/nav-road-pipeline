# QGIS: DOP20 Golden8 固化成果加载指引

本指引用于快速在 QGIS 中总览 8 条带的 DOP20 候选结果（world candidates）。

## 1) 先生成 merged 层

在仓库根目录运行：

```bat
scripts\run_dop20_merge_golden8.cmd
```

脚本会自动选择 `runs\` 下最新且 8 条带均成功的 DOP20 运行，并在该运行目录下生成：

- `runs\<dop20_run>\merged\dop20_candidates_utm32.gpkg`
- `runs\<dop20_run>\merged\dop20_roi_utm32.gpkg`（若存在 ROI）
- `runs\<dop20_run>\merged\dop20_index.csv`

## 2) 在 QGIS 中加载 merged candidates

1. 打开 QGIS
2. 菜单选择：图层 -> 添加图层 -> 添加矢量图层
3. 选择 `dop20_candidates_utm32.gpkg`
4. 图层样式中按 `drive_id` 分类渲染（Categorized）

建议：
- 线框显示（无填充、细边线）更适合叠加其它证据
- CRS 保持为 EPSG:32632（项目坐标系建议同 EPSG:32632）

## 3) 可选：叠加单条带影像与 ROI

若需要更细致检查，可从 `dop20_index.csv` 定位每条带的 mosaic 与 ROI：

- `drives\<drive_id>\evidence\dop20_mosaic_utm32.tif`
- `drives\<drive_id>\roi\roi_buffer100_utm32.gpkg`

按需叠加即可。


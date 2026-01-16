# TOPOISSUE_SPEC（TopoIssue 上报规范：投产必备）

## 1. 目标
当 M9（Validate & Auto-Repair）无法自动修复拓扑问题时，必须输出 TopoIssue：
- 供人工终裁修正
- 供回灌（规则/样例库/训练数据）

## 2. TopoIssue 最小字段（必须）
- issue_id：唯一ID（建议 run_id + 序号）
- window_id / tile_id：定位到窗口或条带
- involved_edges：涉及 edge_id 列表
- involved_nodes：涉及 node_id 列表
- rule_failed：失败的规则名称/编号（例如 CrossLevelConnect、DanglingEnd、WrongDirection）
- severity：S1 / S2 / S3
- description：一句话说明问题
- recommend_actions：推荐人工动作（1~3条）
- evidence_summary：
  - traj_support：轨迹支持度（high/med/low/unknown）
  - lidar_support：点云支持度（high/med/low/unknown）
  - image_support：图像支持度（high/med/low/unknown）
  - prior_conflict：先验冲突提示（none/low/med/high）
  - alignment_residual：对齐残差（如有）
- output_links（可选）：富媒体证据文件名/内部路径引用（投产可为空）

## 3. 严重度定义（默认）
- S1：可接受的轻微问题（不阻断通行/不影响主结构），可批量修
- S2：明显影响几何或局部连通，需要人工关注
- S3：阻断上线（跨层误连、方向错误、关键路口不连通、严重断裂/悬挂）

## 4. 推荐动作字典（建议）
- merge_endpoints：合并端点/提高吸附
- split_edge：重新切分边（路口面内切分）
- remove_spur：删除短刺/悬挂碎边
- reassign_direction：修正方向/单行属性
- prevent_crosslevel_connect：禁止跨层连接，改为上报
- adjust_intersection_polygon：调整路口面范围后再建图
- disable_or_downweight_prior：先验降权/禁用并复跑

## 5. 回灌要求（必须）
每次人工终裁后，必须记录：
- fix_diff（修正前后差分）
- error_code（对应 ERROR_DICT.yaml 的错误类别）
- 将该窗口加入 Stress 集（防回归）
- 更新 M9 规则/阈值（提升 autofix_rate）

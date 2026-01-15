# EVAL_PROTOCOL（RunCard / SyncPack / Gate / 产物协议）

## 1. 目标
评测必须：
- 可复现（同 config_id + data_id + prior_data_id）
- 可对比（Arm0–Arm3）
- 可审计（RunCard/SyncPack 字段稳定）
- 可自动化（Gate PASS/FAIL）

## 2. 每次运行必须产物（runs/<run_id>/）
- StateSnapshot.md
- RunCard_Arm0.md .. RunCard_Arm3.md
- SyncPack_Arm0.md .. SyncPack_Arm3.md
（拓扑启用时加 TopoIssues.*）

## 3. RunCard 最小字段要求
RunCard 必须包含：
- run_id
- arm（Arm0/1/2/3）
- config_id
- runtime_target（windows-native）
- gate（PASS/FAIL）与 gate_reason
- metrics（至少 C/B/A + 先验指标）
- data_summary（drive_count、total_lidar、image_coverage、pose_coverage、index_used、missing_pose_drives 等）
- priors 状态（osm_layers/sat_tiles 是否可用）

### 3.1 指标字段（最小集合）
- C
- B_roughness
- A_dangling_per_km
- prior_used（NONE/OSM/SAT/BOTH）
- prior_confidence_p50
- alignment_residual_p50
- conflict_rate
- image_coverage
- pose_coverage（仅信息输出，不应导致 Gate 假失败）

拓扑启用时（投产目标）：
- topo_pass_rate
- topo_autofix_rate
- topo_issue_rate
- TopoIssue 严重度分布

## 4. SyncPack 字段要求（Diff + Evidence + Ask）
- Diff：相对 baseline 的变更（模块 impl/参数/先验开关）
- Evidence：RunCard 摘要 + TopK 失败窗口/TopoIssue 摘要
- Ask：下一步动作建议（<=3）

## 5. Gate 规则来源
Gate 数值门槛在 configs/gates.yaml：
- C_min
- B_max_roughness
- A_max_dangling_per_km
拓扑门槛在 topo_gate（enabled 时生效）。

要求：
- Gate 必须机器可判定（PASS/FAIL）
- FAIL 必须给 gate_reason

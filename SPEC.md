# 工程提示词（SPEC）：导航电子地图道路生产产线（全 Windows，自动化模块对比与定型，拓扑升级+人机闭环）

## 0. 项目定位与角色
你是“导航电子地图/HD Map 产线架构师 + GIS/点云视觉算法工程师”。本项目目标是在 Windows 原生运行时下，先跑通 POC 端到端闭环，再升级为可投产的拓扑级产线；并长期保留自动化迭代能力：自动对比每个模块的多种实现方案、自动搜索最优组合与参数，按 Gate 规则阶段性冻结为生产线版本；后续定期扫描新论文/新模型再纳入同一机制验证。

本仓库约束：
- 全 Windows（windows-native）为唯一运行时目标。
- PowerShell 可能禁用 ps1，优先使用 scripts/*.cmd 作为一键入口。
- 评测产物协议稳定（RunCard / SyncPack / TopoIssue），不得随意变更字段。

## 1. 总目标（必须实现）
1) 自动化对比与定型：在已定义 M0–M9 模块前提下，自动尝试不同算法/模型实现与参数，找到最优组合并冻结为阶段性生产线版本。
2) 投产链路不依赖外部大模型（禁止产线推理链路调用外部 LLM）。研发阶段可用大模型总线辅助，但产线只吃冻结结果。
3) 输出三类矢量成果，并逐步升级：
- 道路骨架：道路级中心线，方案2（双向分离中心线）
  - POC：输出 polyline 集合（不要求显式图）。
  - 投产：输出拓扑级 Graph（Node/Edge），可校验、可自修复，不能修复输出 TopoIssue 供人工终裁。
- 道路面：drivable polygon（人绘风格：平滑、规整、几何一致）。
- 路口面：intersection polygon（规整，与道路面/骨架连接关系清晰）。
4) 指标优先级：C（完备性） > B（几何） > A（拓扑）。A 为底线，投产必须拓扑自洽+自修复+上报机制。

## 2. 运行时与工程硬约束（全 Windows）
- runtime_target = windows-native（必须写入 RunCard/SyncPack 与冻结包）。
- 依赖可复现：建议 conda-forge 为主并锁版本；冻结包必须包含环境锁定文件。
- 并行按 Windows spawn 编写（入口保护、可 pickle、避免共享大对象）。
- 路径编码统一：使用 pathlib，不写死盘符；通过配置/环境变量注入。
- 网络依赖离线化：OSM/卫星影像必须快照输入（prior_data_id），不得在线实时拉取。
- runs/、cache/、data/、权重、影像、点云等大文件禁止提交 GitHub（仅本地存放）。

## 3. 证据优先级（裁决规则）
主证据（最高优先级）：
- 点云（融合 Tile/单帧）
- 车载视频（L1：语义贴回点云用于净化与补漏）
- 轨迹（投产多车轨迹为强证据；POC可用单车轨迹模拟）

辅助先验（soft prior，可选输入）：
- OSM 快照（道路骨架/边界要素）
- 卫星/航片正射影像快照

裁决原则：
- 先验只能“拉一把”（补漏或几何微调），不能推翻主证据。
- 冲突时自动降权/禁用，并记录 conflict_rate 与禁用比例。
- 对齐残差大或时效差时先验仅用于 QC 展示或禁用。

## 4. 数据与场景
- POC：KITTI-360（7~8 个 drive 条带）。每条带=Tile（交付边界）；条带内可滑窗计算。
- 投产：内部数据目录与 POC 不一致；通过 Adapter 自读取解析。
- 本仓库要求：任何路径/命名规则只允许出现在 Adapter，主流程仅依赖 TileBundle/CIR。

## 5. 统一中间数据合同（CIR / TileBundle）
主流水线只认 TileBundle：
- tile_id / extent / crs（米制）
- pc_tile
- frames[]：timestamp、pose、image_handle、calib_id（可稀疏抽帧）
- trajectories[]：polyline（方向/权重可选）
- priors（可选）：OSM/航片快照引用与元信息

可追溯字段（写入 RunCard/SyncPack）：
- data_id、prior_data_id（如有）、code_version、config_id、model_versions、run_id、runtime_target

## 6. 模块化主流水线（M0~M7，POC先跑通）
说明：条带=Tile 为交付边界；条带内滑窗（200–400m + overlap）为计算边界（参数可调）。

- M0 先验生成与对齐（OSM/卫星统一入口，可降级）
  输出提示要素，不直接生成最终结果：
  road_prior_mask、centerline_hint_lines、boundary_hint_lines、intersection_hint_polygons
  alignment_residual、freshness_score、prior_confidence、conflict_rate
- M1 证据聚合
- M2 drivable 候选（高召回，C优先；先验仅在主证据弱时补漏且受 prior_confidence 控制）
- M3 矢量化（道路面 polygon）
- M4 骨架生成（双向分离中心线 polyline，轨迹为主证据；先验仅提示不决策）
- M5 路口面生成
- M6 几何形态优化（可替换可调参）
  - M6a 道路面优化：去毛刺/填洞/简化/曲率正则/宽度合理性；可用边界提示软吸附（限制最大偏移）
  - M6b 中心线优化：简化+平滑+端点吸附+面内约束（双向一致）
  - M6c 路口面优化：规整、拼接顺滑、稳定性提升；可用路口提示软约束
- M7 拓扑底线修复（POC阶段）

## 7. 拓扑升级层（投产必须：Graph + 自修复 + 上报）
- M8 Graph Build：polyline → Node/Edge Graph（端点聚类、路口切分、方向一致、跨层不乱连）。
- M9 Validate & Auto-Repair：检查→修复→再检查；失败则输出 TopoIssue。

TopoIssue 最小字段：
- issue_id
- window_id/范围
- edge_id/node_id 列表
- 失败规则 + 严重度（S1/S2/S3）
- 推荐动作
- 主证据支持度 + 先验冲突提示（仅参考）

人工终裁回灌（三条必须落地）：
1) 规则/参数回灌（提升 M9 修复率）
2) 样例库回灌（加入 Stress 集防回归）
3) 训练数据回灌（如需微调语义/路口面/拆分）

## 8. 图像使用策略（先 L1）
- L1：语义贴回点云（净化+补漏）
- L2/L3：后续可重启（标线/隔离、稠密重建）

## 9. 评估体系（双通道输出 + 机器 Gate）
主指标（产线可用，不依赖外部真值）：
- C：轨迹覆盖、走廊覆盖、双向匹配率
- B：毛刺度/折点密度、曲率异常、宽度平滑、重叠区稳定性
- A：底线违规统计

先验指标（必须记录）：
- prior_used（None/OSM/SAT/BOTH）
- prior_confidence 分位数、alignment_residual、conflict_rate

拓扑指标（投产必须）：
- topo_pass_rate、topo_autofix_rate、topo_issue_rate、严重度分布

产物协议（每次 run 必须在 runs/<run_id>/ 输出）：
- StateSnapshot.md
- RunCard_Arm0..3.md
- SyncPack_Arm0..3.md
（拓扑启用时加 TopoIssues.*）

## 10. 四臂消融（标准化、长期固定）
- Arm0：Base（点云+视频L1+轨迹）
- Arm1：Base + OSM
- Arm2：Base + SAT
- Arm3：Base + OSM + SAT
每次迭代至少跑 Golden 回归集，输出每臂独立 RunCard/SyncPack。

## 11. SyncPack 文本协议（唯一同步桥梁）
每次运行必须生成：
- StateSnapshot.md
- RunCard_Arm*.md
- SyncPack_Arm*.md
（投产启用 M8/M9 时加 TopoIssue 摘要）

SyncPack 必含：Diff（配置差分）+ Evidence（指标+TopK窗口+TopoIssue摘要）+ Ask（下一步建议）。

## 12. 错误类型字典（<=12类）与责任模块映射
必须稳定，写入 ERROR_DICT.yaml，用于自动归因与 TopK 展示。

## 13. 长期迭代与定型发布（Lab/RC/Prod 三泳道）
- Lab：允许快速换模块/换总线；自动化搜索与对比在此进行。
- RC：受控变更，完整回归+抽检，准备定型。
- Prod：冻结包运行（禁外部LLM），可回滚。

冻结包必须包含：
code_version / config_id / model_versions / prior_data_id / benchmark_id / RunCard+SyncPack / runtime_target

## 14. Adapter 自读取（POC/投产目录不一致）
- 任何路径/命名规则只允许出现在 Adapter。
- 程序从根目录自动发现数据、建立索引、关联 pose-图像-点云-标定-轨迹-先验快照。
- 缺失数据降级运行并在 RunCard/SyncPack 标注完整性。

## 15. 自动化运行合同（必须提供的入口）
仓库根目录必须提供至少四个一键入口（Windows cmd）：
- scripts/setup.cmd
- scripts/smoke.cmd
- scripts/eval.cmd（支持 --config/--max-frames 等参数）
- scripts/autotune.cmd
- scripts/index.cmd（生成 cache/kitti360_index.json）

## 16. 默认：模块注册表与自动化对比验证（按默认策略）
### 16.1 模块注册表（modules_registry.yaml）
每个实现条目必须包含：
- module、impl_id、impl_version、description
- runtime_target=windows-native、device、min_gpu_mem_gb、expected_cost（粗略）
- deps、artifacts.weights（本地/内部路径）、artifacts.license（必填）
- param_schema（name/type/default/range/choices/scale/tunable）
- requires（priors_osm/priors_sat/trajectories/images）、degrade_policy
- io_contract.inputs/outputs
- status（stable/candidate/deprecated）、added_date

强制：不得在线实时拉取 OSM/影像；必须快照输入 prior_data_id。

### 16.2 autotune 默认三阶段（A筛选→B联合→C定型）
Stage A：单模块快速筛选（固定其它模块为 baseline，仅替换一个模块实现；数据用 Golden 子集+Stress 少量窗口；保留 Top-2）
Stage B：联合组合搜索（Top-2 空间内联合“实现选择+参数调优”；Budget 默认 20–50 trial；多保真+早停）
Stage C：定型验证（Top-3 全量回归 Golden+Refresh+Stress；跑 Arm0–Arm3；生成 winner_active.yaml 候选冻结）

产物：
- stageA_topk.json、trials_top.json、leaderboard.md、winner_hint.md、winner_active.yaml
- 每个 trial 至少有可解析的结果摘要

## 17. 现阶段工程状态（仓库现实）
- eval_all 已支持真实 KITTI-360 数据评测（Arm0–Arm3）
- index cache 已生成并可加速 eval（cache/kitti360_index.json，不提交）
- autotune 已通过 sim_metrics 与 eval_all 解耦（避免真实 eval 改动影响搜索机制）
- 后续真实 autotune 将逐步替换 sim_metrics 为真实多保真评测（先小样本/限帧，再全量）


## 18. Phase Milestones (Stage Summary)

### Current Phase: Crosswalk POC (Delivered)
- fullpose projection chain
- v4 lidar support loop (accum lidar-fit)
- robust IoU metrics (bbox + dilated)
- Stage2 continuous inference (SAM2 video) with ROI + drift control
- strict frame-range assertions
- autotune framework (short range + long range)

### Next Phase: Architecture Consolidation
- validate "intersection + crosswalk" feature outputs on the unified architecture

## 19. Submission Boundary (This Phase)
- Includes: documentation upgrades + current runnable script entrypoints
- Excludes: runs/ cache/ weights (no large artifacts)
- Minimal reproducibility path: setup -> smoke -> eval -> crosswalk strict regression

## 20. ????????????????
- ?? schema ?????????backend_status / fallback_used / fallback_from / fallback_to / backend_reason
- ?????
  - scripts/run_image_providers.cmd
  - scripts/run_ab_eval.cmd
  - scripts/setup_sam3_weights.cmd
- STRICT_BACKEND=1 ?? fallback??????? fallback ?????

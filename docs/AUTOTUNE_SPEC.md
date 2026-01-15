## 1. 目标
在 M0–M9 已定义的前提下，自动化完成：
- 模块实现方案对比（算法/模型/规则）
- 组合搜索（实现选择 + 参数调优）
- 阶段性定型（冻结为生产候选 config）
并输出可复现证据：leaderboard + winner + SyncPack。

## 2. 输入与依赖（单一事实源）
- modules_registry.yaml：模块实现注册表
- configs/active.yaml：当前 baseline 冻结配置
- configs/arms.yaml：Arm0–Arm3 先验开关
- configs/gates.yaml：Gate 门槛
- configs/search.yaml：预算（TopK、trial 数、TopN）
- cache/kitti360_index.json：索引缓存（不进仓库，用于加速与抽样）

## 3. 四臂消融（固定）
- Arm0: Base
- Arm1: Base + OSM
- Arm2: Base + SAT
- Arm3: Base + OSM + SAT
规则：任意 Arm Gate FAIL 可早停该 trial（Prune）。

## 4. 默认三阶段策略
### Stage A：单模块快速筛选（Screening）
- 固定其它模块为 baseline，仅替换某一模块 impl
- 小样本评测（建议用 index cache + max_frames=500~2000）
- 默认只跑 Arm0；若模块依赖先验则补跑对应 Arm
- 保留 TopK（默认 K=2），输出 stageA_topk.json

### Stage B：联合组合搜索（Joint Search）
- TopK 空间内联合“实现选择 + 参数调优”
- 预算 stageB_budget_trials（默认 20，可调 20~50）
- 多保真：先小样本再大样本
- 早停：任意 Arm Gate FAIL 即停
- 输出 trials_top.json 与 leaderboard.md

### Stage C：定型验证（Freeze）
- 对 TopN（默认 N=3）扩大评测（更大 max_frames / 更多 drive；后续再接 Golden/Refresh/Stress）
- 跑 Arm0–Arm3 全套
- 产出 winner_active.yaml 与 winner_hint.md（不直接覆盖 active）

## 5. 产物约定
每次 autotune 在 runs/<run_id>/ 产出：
- stageA_topk.json
- trials_top.json
- leaderboard.md
- winner_active.yaml（若产生 winner）
- winner_hint.md（若产生 winner）
说明：runs/ 不提交 GitHub；候选冻结配置复制到 configs/candidates/ 后再提交。

## 6. 新论文/新模型接入（后续）
- ResearchScout 发现新候选 → 写入 registry（status=candidate）
- 触发 Stage A → 入 TopK 再进 Stage B → 达标再 Stage C 定型
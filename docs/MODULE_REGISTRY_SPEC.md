# MODULE_REGISTRY_SPEC（modules_registry.yaml 规范）

## 1. 目标
为 M0–M9 每个模块登记多种实现（算法/模型/规则），以支持：
- Stage A 单模块筛选（Screening）
- Stage B 组合搜索 + 参数调优（Joint Search）
- Stage C 定型冻结（Freeze）
- 可追溯、可复现、可回滚

## 2. 文件位置与原则
- 注册表文件：仓库根目录 modules_registry.yaml
- 强制原则：
  1) runtime_target=windows-native
  2) OSM/卫星影像必须以快照输入（prior_data_id），禁止在线实时拉取
  3) 权重/大文件不进仓库，只能本地存放并通过路径引用
  4) 必须声明许可证（artifacts.license）

## 3. 每个实现 entry 必填字段
- module / impl_id / impl_version / description
- runtime_target / device / min_gpu_mem_gb / expected_cost（推荐）
- deps（推荐）与 artifacts.weights（本地路径）+ artifacts.license（必填）
- param_schema（可调参数空间）
- requires（是否依赖 OSM/SAT/轨迹/图像）与 degrade_policy（缺失降级策略）
- io_contract.inputs / io_contract.outputs（输入输出契约）
- status（stable/candidate/deprecated）与 added_date

## 4. 示例（示意）
```yaml
implementations:
  - module: M2
    impl_id: m2_stub
    impl_version: 0.0.1
    description: "Baseline drivable candidate (stub)"
    runtime_target: windows-native
    device: cpu
    min_gpu_mem_gb: 0
    artifacts: {weights: null, license: "N/A"}
    param_schema:
      - {name: dummy_thr, type: float, default: 0.5, range: [0.1, 0.9], scale: linear, tunable: true}
    requires: {priors_osm: false, priors_sat: false, trajectories: false, images: false}
    degrade_policy: {on_missing: disable}
    io_contract: {inputs: [pc_tile], outputs: [drivable_candidate_mask]}
    status: stable
    added_date: "2026-01-15"
5. 合规提醒

OSM 涉及 ODbL 合规；仓库只存快照版本标识与结果描述，不存敏感/受限数据。

第三方模型权重/代码必须标明 license，避免投产合规风险。

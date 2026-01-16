# AGENTS.md — Codex/Agent Guardrails (Windows-native)

本项目运行时目标：windows-native（全 Windows）。任何实现必须遵守以下规则。

## 1) 单一事实源与必须读取的文件
- 必须优先阅读：SPEC.md、docs/EVAL_PROTOCOL.md、configs/*.yaml、modules_registry.yaml
- 不得修改 SPEC.md，除非用户明确要求

## 2) 运行入口与执行策略
- PowerShell 可能禁止运行 ps1；优先使用 .cmd 脚本：
  - .\scripts\setup.cmd
  - .\scripts\smoke.cmd
  - .\scripts\eval.cmd
  - .\scripts\autotune.cmd
  - .\scripts\index.cmd
- 每次修改后必须执行（至少）：
  1) .\scripts\smoke.cmd
  2) .\scripts\eval.cmd --max-frames 2000
- 如涉及索引：先 .\scripts\index.cmd --max-frames 2000 再 eval

## 3) 产物协议（强约束）
每次评测必须在 runs\<run_id>\ 输出：
- StateSnapshot.md
- RunCard_Arm0..3.md
- SyncPack_Arm0..3.md
（拓扑启用时再加 TopoIssue/TopoIssues）

输出字段必须保持稳定可解析（RunCard/SyncPack 格式不随意变更）。

## 4) 禁止提交与敏感规则
- 禁止提交：runs/、cache/、data/、大文件（点云/影像/视频/权重）
- 任何下载的权重/数据只允许放本地，并通过路径/快照版本号引用
- 不要把任何密钥写入仓库（API key、token 等）

## 5) Git 工作流（可回滚）
- 修改前建议：git status 确保干净；必要时先提交 checkpoint
- 修改后：只提交代码/配置/文档，不提交 runs/cache/data
- commit message 要明确（feat/fix/chore/docs）且描述变更目的

## 6) Windows 并行约束
- 多进程必须按 Windows spawn 方式编写：
  - if __name__ == '__main__' 保护
  - 子进程参数可 pickle
  - 避免传递大对象

## 7) 目标优先级（写入决策）
- C（完备性） > B（几何） > A（拓扑）
- 点云+车载视频为最高优先级证据；OSM/航片为 soft prior，冲突时降权/禁用并记录 conflict_rate

## 8) 自动化演进（默认策略）
- autotune 按 Stage A/B/C 执行：筛选 -> 联合搜索 -> 定型
- 新模块/新模型必须先过契约测试与小样本评测，再进入联合搜索

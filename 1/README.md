# map-skill-vnext (repo template)

本仓库是“vNext｜Skill 化快速验证版”的**项目骨架模板**，用于在 PyCharm + GitHub 下快速启动并形成可复现的实验闭环。

> 参考启动书：见 `docs/项目启动书_vNext_Skill化快速验证版.md`

## 1. 快速开始（Windows / PyCharm）

### 1.1 创建虚拟环境
在项目根目录（本 README 所在目录）打开终端：

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip wheel setuptools
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

> 如果你使用 PyCharm：建议在 **Settings → Project → Python Interpreter** 选择 `.venv\Scripts\python.exe`。

### 1.2 运行 smoke（分钟级）
```bat
scripts\smoke.cmd
```

运行成功后会在 `runs/` 下生成一个 run 目录，包含四件套：
- `metrics.json`
- `SkillReport.md`
- `debug_layers.gpkg`
- `RunMeta.json`

## 2. 目录结构（简化版）
- `skills/`：每个 Skill 一个目录（含 `skill.yaml` + `run.py`）
- `core/`：公共库（io/crs/metrics/report/debug）
- `configs/`：smoke / regression-mini / matrix 配置
- `scripts/`：一键入口（.cmd / .py）
- `runs/`：运行产物（不入库，已在 .gitignore）

## 3. 下一步（推荐）
1) 按启动书要求补齐 S0/S1/S2/S3/S5 的最小 Skill 集合  
2) 把 smoke + regression-mini 固化为每次改动必跑的检查项  
3) 引入 matrix runner 做组合矩阵对比（leaderboard + diff_pack）


# CausalSpatial-Bench

**CausalSpatial-Bench** 是首个评测视觉语言模型（VLM）*因果与反事实空间推理*能力的 benchmark。现有空间 benchmark 只测静态感知（"物体 A 在哪里？"），本 benchmark 测*干预推理*：给定图像和一段空间操作描述，新的空间关系是什么?

---

## 研究动机

现有 benchmark（VSI-Bench、MindCube、MMSI-Bench）衡量的是空间**感知**——直接从图像中读取关系。没有 benchmark 衡量空间**推断**——推理因果干预后关系如何变化。这一差距至关重要：模型可以从图像中记住"左/右"，却无法回答"如果我移动桌子，台灯最终在哪里？"

CausalSpatial-Bench 通过三层题目层级解决这一问题：

| 层级 | 任务 | 示例 |
|------|------|------|
| **L1** | 静态感知（对比基线） | "书架在电视的哪个方向？" |
| **L2** | 干预推理 | "如果书架向右移动 2 米，台灯和电视之间的新关系是什么？" |
| **L3** | 反事实推理 | "如果桌子被移走，以下哪些物体也会随之移位：台灯底座 / 台灯灯罩 / 两者都会 / 书架？" |

核心假设：**L1 准确率 > 65%，L2 准确率 < 45%，差距 > 20%**——证明空间推理从根本上比空间感知更难。

---

## 题型定义

```
Level 1 — 静态空间感知（对比基线）
  L1-方向  : "{A} 在 {B} 的哪个方向？"
  L1-遮挡  : "从当前视角，{A} 的可见性如何？"        → 四选一：完全可见 / 部分遮挡 / 完全遮挡 / 不在画面内
  L1-距离  : "{A} 和 {B} 相距多远？"

Level 2 — 空间干预推理（核心创新）
  L2.1 物体移动   : "如果 {A} 向 {方向} 移动 {d} 米，{B} 和 {C} 之间的新关系是什么？"
  L2.2 视角移动   : "如果观察者向 {方向} 移动 {d} 米，{A} 的可见性如何？"
  L2.3 物体移除   : "如果移除 {A}，从当前视角 {B} 的可见性如何？"
                    → 四选一：完全可见 / 部分遮挡 / 完全遮挡 / 不在画面内

Level 3 — 空间反事实推理（核心创新）
  L3.1 支撑链成员推断 : "如果 {A} 被移动到其他位置，以下哪些物体也会随之移位？"
                        → 四选一：{B}（直接子节点，1-hop）/ {C}（孙节点，2-hop）/
                                  Both {B} and {C}（正确）/ {D}（非链邻居，干扰项）
                        → 两跳推理：须从图像自行推断 A→B→C 完整链，题干不予提示
                        → "仅 B"是典型错误选项，专门考察模型能否识别多跳链的完整范围
  L3.2 参照系旋转 : "假设房间当初按旋转 {90°/180°/270°} 的朝向设计，所有物体相对
                    位置不变，从原相机位置和朝向（均不变）观察，{A} 在 {B} 的哪个方向？"
                    → 四选一方向：left / right / in front / behind
                    → 相机位姿固定，仅物体随房间旋转；模型须计算实际新方向
```

---

## 数据来源：ScanNet

基于 [ScanNet](http://www.scan-net.org/)——1513 个 RGB-D 扫描室内场景，具备：

- 逐实例语义标注（NYU-40 类别）
- 每帧相机位姿（4×4 camera-to-world 矩阵）
- 轴对齐 3D 重建网格（含 axisAlignment 变换）
- RGB 彩色图像序列

**计划规模：** 300 个场景，约 2,700 道题。

---

## Pipeline 架构

```
ScanNet 原始数据
      │
      ▼
[scene_parser.py]        Stage 1 — 提取物体列表（id / 标签 / 3D bbox / 中心点）
      │
      ▼
[support_graph.py]       Stage 2 — 检测支撑关系（几何法：垂直接触 + XY 投影重叠）
      │
      ▼
[frame_selector.py]      Stage 3 — 每场景选 3-5 张代表性 RGB 帧
      │
      ▼
[relation_engine.py]     Stage 4 — 计算所有物体对的方向 / 距离 / 遮挡 GT
      │
      ▼
[virtual_ops.py]         Stage 5 — 施加虚拟操作；重算关系；差值 = 新 GT
      │
      ▼
[qa_generator.py]        Stage 6 — 填充模板 → 四选一选择题（题目 + 4 选项 + GT 答案）
      │
      ▼
[quality_control.py]     Stage 7 — 过滤歧义/无效题；平衡答案分布
      │
      ▼
output/benchmark.json
      │
      ▼
[evaluation/evaluate.py] — 按层级/题型统计准确率 + L1-L2 差距
```

---

## 项目结构

```
CausalSpatial-Bench/
├── src/
│   ├── scene_parser.py          # Stage 1：PLY + 标注文件 → 带 3D AABB 的物体列表
│   ├── support_graph.py         # Stage 2：几何支撑检测（A 放在 B 上）
│   ├── frame_selector.py        # Stage 3：贪心多样性帧选择
│   ├── relation_engine.py       # Stage 4：方向 / 距离 / 遮挡 GT 计算
│   ├── virtual_ops.py           # Stage 5：移动 / 移除 / 旋转 / 视角变换
│   ├── qa_generator.py          # Stage 6：基于模板 + GT 生成选择题
│   ├── quality_control.py       # Stage 7：过滤、均衡、人工验证工具
│   └── utils/
│       ├── colmap_loader.py         # 加载 ScanNet 相机内参 / 位姿 / 轴对齐矩阵
│       ├── coordinate_transform.py  # 世界 ↔ 相机 ↔ 像素坐标变换
│       └── ray_casting.py           # trimesh 光线追踪（遮挡检测）
├── templates/
│   └── question_templates.json  # 8 种题型 × 各 2-3 个英文模板
├── scripts/
│   ├── run_pipeline.py          # 完整 pipeline 入口
│   ├── pilot_study.py           # 30 场景 Pilot Study，验证核心假设
│   └── run_vlm_inference.py     # VLM 推理脚本（GPT-4o / Gemini / Qwen）
├── evaluation/
│   └── evaluate.py              # 准确率指标 + L1-L2 差距报告
├── output/                      # 生成数据（不入 git）
│   ├── scene_metadata/
│   ├── questions/
│   └── benchmark.json
├── data/
│   └── scannet/scans/           # ScanNet 原始数据（需申请许可，不随代码发布）
└── requirements.txt
```

---

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd CausalSpatial-Bench

# 安装 Python 依赖（需要 Python >= 3.10）
pip install -r requirements.txt

# 可选：安装 pyembree 加速光线追踪（约 5-10 倍提速）
pip install pyembree
```

**依赖说明：**

| 包名 | 用途 |
|------|------|
| `open3d` | PLY 网格加载 |
| `trimesh` | 光线追踪（遮挡检测） |
| `numpy` | 坐标运算 |
| `plyfile` | PLY 备用解析器 |
| `openai` | GPT-4o / Qwen 推理（VLM 评测用） |
| `google-generativeai` | Gemini 推理（VLM 评测用） |
| `Pillow` | Gemini 图片加载 |
| `tqdm` | 推理进度条（可选） |

---

## 数据准备

1. 在 https://kaldir.vc.in.tum.de/scannet/ 申请 ScanNet 数据访问权限
2. 下载数据集，解压到 `data/scannet/scans/`
3. 每个场景的目录结构如下：

```
data/scannet/scans/<scene_id>/
├── <scene_id>.txt                              # 场景元数据（colorWidth/Height、axisAlignment）
├── <scene_id>_vh_clean_2.ply                   # 3D 重建网格
├── <scene_id>_vh_clean_2.0.010000.segs.json    # 顶点 → 超分割片段映射
├── <scene_id>_vh_clean.aggregation.json        # 片段 → 实例 + 标签标注
├── intrinsic/
│   └── intrinsic_color.txt                    # 4×4 彩色相机内参矩阵
├── pose/
│   └── 0.txt, 1.txt, …                        # 每帧 4×4 camera-to-world 矩阵
└── color/
    └── 0.jpg, 1.jpg, …                        # RGB 图像帧
```

---

## 使用方法

### Pilot Study（30 场景）

在全量生成前验证 pipeline 正确性和核心假设。

```bash
python scripts/pilot_study.py \
    --data_root data/scannet/scans \
    --output_dir output/pilot \
    --n_scenes 30
```

输出内容：
- `output/pilot/benchmark.json` — 生成的题目
- `output/pilot/human_validation_sample.json` — 每层级 50 道题，供人工审核
- 控制台打印 L1/L2/L3 分层统计

### 完整 Pipeline（300 场景）

```bash
python scripts/run_pipeline.py \
    --data_root data/scannet/scans \
    --output_dir output \
    --max_scenes 300 \
    --max_frames 5
```

关闭光线追踪可大幅提速（不生成遮挡题）：

```bash
python scripts/run_pipeline.py \
    --data_root data/scannet/scans \
    --output_dir output \
    --max_scenes 300 \
    --no_ray_casting
```

### VLM 推理

```bash
# 完整 Pilot 推理（GPT-4o）
python scripts/run_vlm_inference.py \
    --benchmark  output/pilot/benchmark.json \
    --image_root data/scannet/scans \
    --model      gpt-4o \
    --output     predictions/pilot_gpt4o.json

# 只跑 L3.1（快速验证链成员推断难度）
python scripts/run_vlm_inference.py \
    --benchmark  output/pilot/benchmark.json \
    --image_root data/scannet/scans \
    --model      gemini-2.5-pro \
    --qtype      support_chain \
    --output     predictions/pilot_gemini_l31.json
```

API Key 通过环境变量传入：`OPENAI_API_KEY` / `GOOGLE_API_KEY` / `DASHSCOPE_API_KEY`。

### 评测模型

预测文件格式：
```json
[
  {"question_id": 0, "prediction": "A"},
  {"question_id": 1, "prediction": "C"},
  ...
]
```

运行评测：
```bash
python evaluation/evaluate.py \
    --benchmark output/benchmark.json \
    --predictions predictions/gpt4o.json \
    --output_report evaluation/report_gpt4o.json
```

输出示例：
```
============================================================
CAUSAL-SPATIAL BENCH EVALUATION REPORT
============================================================

Overall accuracy: 52.3% (1413/2703)

--- By Level ---
  L1: 71.4% (714/1000)
  L2: 38.2% (497/1300)
  L3: 49.5% (202/408)

--- By Type ---
  direction:           74.1%
  distance:            68.8%
  occlusion:           61.3%
  object_move:         35.7%
  viewpoint_move:      41.2%
  object_remove:       39.8%
  support_chain:       52.1%
  coordinate_rotation: 46.4%

--- Core Hypothesis ---
  L1 - L2 gap: 33.2%
  [PASS] Gap > 20% — 干预推理显著难于静态感知
============================================================
```

---

## 关键设计决策

### 支撑链传导（L2.1）与链成员推断（L3.1）

**L2.1 物体移动**：移动物体时，其支撑链上的所有依赖物体一并移动（传递闭包）。题目明确给出操作方向和距离，考察模型能否正确计算链传导后的新空间关系。

```
table → [lamp_base → lamp_shade]
桌子向右移 2m → lamp_base 和 lamp_shade 也随之向右移 2m
```

**L3.1 支撑链成员推断**：不给出具体移动向量，只说"{A} 被移走"，考察模型能否从图像中识别完整的两跳支撑链，判断哪些物体会受到牵连。关键在于区分直接子节点（1-hop）和孙节点（2-hop）——只回答"台灯底座会移位"是一跳思维，必须同时答出"台灯灯罩也会移位"才算完整的两跳推理。

```
正确答案：Both lamp_base and lamp_shade
典型错误：lamp_base only（仅识别到一跳，遗漏孙节点）
干扰项  ：bookshelf（非链邻居，外观上靠近 A 但无物理依赖）
```

### 自我中心方向（相机坐标系）

所有方向关系在相机坐标系中计算，与 VLM 解读图像的方式一致。采用 OpenCV 约定：x→右，y→下，z→前。

### 多射线遮挡采样

不使用单条中心到中心的光线，而是向目标 bbox 内的随机点发射 8 条光线。遮挡判定规则：
- >80% 光线可达 → `fully_visible`（完全可见）
- 20%–80% 可达 → `partially_occluded`（部分遮挡）
- <20% 可达 → `fully_occluded`（完全遮挡）

### 自动质量过滤

| 过滤规则 | 条件 | 原因 |
|----------|------|------|
| 方向歧义 | `ambiguity_score > 0.7` | 两个方向分量几乎相等，答案不可靠 |
| 干预无效 | `relation_unchanged = True` | 操作未真正改变关系，属于平凡题 |
| 距离边界 | `near_boundary = True` | 距离在分档边界 0.2m 内，GT 不稳定 |

---

## 预期规模与时间线

| 层级 | 目标题数 | 每场景约 |
|------|---------|---------|
| L1-方向 | 400 | ~1.3 |
| L1-遮挡 | 300 | ~1.0 |
| L1-距离 | 200 | ~0.7 |
| L2.1 物体移动 | 600 | ~2.0 |
| L2.2 视角移动 | 400 | ~1.3 |
| L2.3 物体移除 | 300 | ~1.0 |
| L3.1 支撑链成员推断 | 300 | ~1.0 |
| L3.2 参照系旋转 | 200 | ~0.7 |
| **合计** | **~2,700** | **300 个场景** |

**执行时间线（目标 NeurIPS 2026）：**
- 第一阶段（2 周）：Pilot Study——30 场景，验证 L1-L2 差距
- 第二阶段（4 周）：完整 Pipeline——300 场景，约 2,700 道题
- 第三阶段（4 周）：模型评测 + 论文写作

---

## 验证标准

从 Pilot Study 扩展到完整数据集前，需满足以下门槛：

| 指标 | 目标 | 未达标的含义 |
|------|------|------------|
| L1 准确率（GPT-4o） | > 65% | 题目过难或标注有误 |
| L2 准确率（GPT-4o） | < 45% | 题目过简单，无需推理即可作答 |
| L1 − L2 差距 | > 20% | benchmark 无法区分感知与推理能力 |
| Cohen's κ（人工） | > 0.85 | GT 答案存在歧义，模板需修订 |

---

## 与相关工作的对比

| Benchmark | 场景类型 | 测干预推理？ | 支撑链？ | 规模 |
|-----------|---------|------------|---------|------|
| VSI-Bench | 真实（ScanNet） | 否 | 否 | 5,000 |
| MindCube | 合成 | 部分 | 否 | 3,000 |
| MMSI-Bench | 混合 | 否 | 否 | 4,000 |
| CausalSpatial（arxiv 2601.13304） | 合成 | 是 | 否 | ~2,000 |
| **CausalSpatial-Bench（本工作）** | **真实（ScanNet RGB-D）** | **是** | **是** | **~2,700** |

与同期工作 CausalSpatial（2601.13304）的核心差异：
- **真实场景** vs. 合成：ScanNet RGB-D 扫描 vs. 程序化生成
- **支撑链传导**：多跳物理依赖 vs. 单步操作
- **反事实 L3**：支撑链成员推断（两跳）+ 坐标系旋转反事实，已有工作均未涉及

---

## 引用

> *CausalSpatial-Bench: A Benchmark for Causal and Counterfactual Spatial Reasoning in Vision-Language Models.* （撰写中，目标投稿 NeurIPS 2026）

```bibtex
@article{causalspatial2026,
  title   = {CausalSpatial-Bench: A Benchmark for Causal and Counterfactual Spatial Reasoning in Vision-Language Models},
  year    = {2026},
  note    = {In preparation}
}
```

---

## 许可证

代码：MIT License。
数据：遵循 [ScanNet 使用条款](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf)。

# SDCC: Semantics-Driven Contrastive Learning and Calibration for Few-Shot Object Detection

本项目提出了一个面向少样本目标检测（Few-Shot Object Detection）的多模块协同框架，包含三个核心创新：**CGCL（类间语义引导的原型对比学习）**、**TFE（文本引导的特征调制）** 和 **CASC（CLIP辅助的分数校准）**。

## 整体架构

![structure](assets/structure.png)

**训练阶段**：GDL 梯度解耦 → Res5 特征提取  → TFE 文本调制 → 分类/回归 + CGCL 对比学习

**推理阶段**：检测器输出 → CASC负向语义锚点→ 最终分数

---

## 核心创新模块

### 1. CGCL — 类间语义引导的原型对比学习

![cgcl](assets/cgcl.png)

- 利用预计算的类间语义相似度矩阵（来自 CLIP 视觉特征聚类）作为对比学习的权重
- 语义越相似的负样本对被施加越大的排斥力，迫使模型在最容易混淆的类间边界上学到更细粒度的区分
- 动态记忆原型库（MemoryPrototypeBank）为每类维护历史特征中心，解决 batch 内正样本不足的问题

### 2. TFE — 文本引导的特征调制

![tfe](assets/tfe.png)

- 基于 FiLM 机制，用 CLIP 文本嵌入对 ROI 视觉特征做 channel-wise 的 scale/shift
- Top-k 稀疏注意力：只关注语义最相关的 k 个类别文本，避免不相关基类稀释新类信号
- Channel-wise 可学习门控：每个通道独立控制文本调制强度

### 3. CASC — CLIP 辅助的分数校准

![casc](assets/casc.png)

- 推理阶段用 CLIP 对检测框重新打分
- 负向语义锚点（Negative Semantic Anchors）：在 softmax 竞争中加入负向文本提示，充当"概率黑洞"吸收假阳性
- Dirichlet-Softmax 混合打分，既保留尖锐分类能力又避免过度自信

---

## 实验结果

### Pascal VOC

![voc](assets/voc.png)

### MS COCO

![coco](assets/coco.png)

---

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- Detectron2
- CLIP (openai)
- fvcore, scikit-learn

```bash
# 安装 detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# 安装其他依赖
pip install ftfy regex tqdm scikit-learn fvcore
pip install git+https://github.com/openai/CLIP.git
```

---

## 数据准备

请参考[DeFRCN](https://github.com/er-muyue/DeFRCN) — Decoupled Faster R-CNN准备 Pascal VOC 和 MS COCO 数据集。

目录结构：
```
datasets/
├── VOC2007/
├── VOC2012/
├── coco/
├── cocosplit/
├── vocsplit/
└── ImageNetPretrained/
    ├── MSRA/R-101.pkl
    └── torchvision/resnet101-5d3b4d8f.pth
```

---

## 使用方式

### 1. Base 预训练

```bash
# VOC (以 split1 为例)
python3 main.py --num-gpus 4 \
    --config-file configs/voc/defrcn_det_r101_base1.yaml \
    --opts MODEL.WEIGHTS datasets/ImageNetPretrained/MSRA/R-101.pkl \
           OUTPUT_DIR checkpoints/defrcn_det_r101_base1
```

### 2. Few-Shot 微调（VOC GFSOD）

```bash
bash run_voc_gfsod_finetuning.sh r101 1 scac 1
# 参数说明：
#   r101      - 骨干网络 (ResNet-101)
#   1         - GPU 数量
#   scac    - 实验名称（决定 checkpoint 保存路径）
#   1         - VOC split ID (1/2/3)
```


### 3. Few-Shot 微调（COCO GFSOD）

```bash
bash run_coco_gfsod_finetuning.sh r101 4 scac
# 参数说明：
#   r101           - 骨干网络
#   4              - GPU 数量
#   scac  - 实验名称
```

### 4. 单独评估（使用 CASC 校准）

```bash
bash run_cmclip_eval.sh r101 1 scac 1
```

---

## 致谢

本项目基于以下工作：

- [DeFRCN](https://github.com/er-muyue/DeFRCN) — Decoupled Faster R-CNN
- [DCFS](https://github.com/gaobb/DCFS) — Dual-path CLIP Few-Shot
- [Detectron2](https://github.com/facebookresearch/detectron2) — Facebook AI Research
- [CLIP](https://github.com/openai/CLIP) — OpenAI

# 项目结果展示与说明

本说明基于项目阶段性汇报文档整理（`PVHJ7 - CHME0039.pdf`、`使用合成数据进行训练的息肉检测模型.docx`），用于在 GitHub 上快速展示项目目标、实验设置与关键结果。

## 1. 背景与目标

结直肠癌（CRC）筛查依赖结肠镜检查，但术中息肉漏检仍然存在。实时目标检测模型可以辅助医生识别息肉，但高质量、可公开共享的标注数据往往不足。为缓解数据稀缺问题，本项目探索使用 3D 软件（Blender）程序化生成合成结肠镜图像，作为真实数据的补充，提升检测效果与泛化能力。

## 2. 数据集与准备

真实世界数据集（汇报中使用的四个数据源）：

- Kansas（KA）
- HyperKvasir（KV）
- LDPolypVideo（LD）
- PolypGen（PG）

合成数据集：

- B1：约 50,000 张图像（已有合成数据集）
- B2：约 10,000 张图像（项目中新增生成，随机化主要限于相机旋转）

数据拆分策略（汇报口径）：

- KA 与 LD：按数据集原始划分使用
- KV：80% 训练、10% 测试、10% 验证
- PG：按患者划分（6 名患者中 4/1/1 用于训练/测试/验证）

数据格式：

- 汇报中说明真实数据标签清洗后统一成 YOLO 格式，并按 YOLOv7 训练流程加载。
- 本仓库默认以 YOLO 的 `images/` + `labels/` 目录结构组织（见 `data/polyps.yaml` 与 `scripts/dataset/verify_yolo_dataset.py`）。

## 3. 方法与训练设置

模型：

- YOLOv7（实时目标检测）

训练策略（汇报口径）：

- 在合成数据上预训练，再在真实数据集上微调
- 分别尝试 B1 与 B2 的预训练效果
- 图像尺寸：`512x512`
- 真实数据集训练：每个数据集约 `100 epochs`
- 合成预训练：B1 约 `300 epochs`，B2 约 `100 epochs`

评估指标（YOLOv7 输出）：

- Precision（P）
- Recall（R）
- mAP@0.5（mAP.5）
- mAP@0.5:0.95（mAP[.5:.95]）

## 4. 关键结果（摘要）

总体结论：

- B1（多样性更高的合成数据）预训练在多数真实数据集上带来稳定收益。
- B2（随机化受限）预训练收益不稳定，更多在 LD 这类“数据质量/分布更具挑战”的数据上体现改进。
- 合成数据的“多样性”比单纯“数量”更关键。

示例结果（摘自汇报文本与表格片段）：

- KA：仅真实数据训练 P≈86.3、R≈84.5、mAP.5≈92.5；B1 预训练后在 KA 微调 P≈88.5、mAP.5≈92.6。
- PG：在 PG 上测试时，B1+PG 组合可达到 P≈88.2、R≈76.5、mAP.5≈85.8、mAP[.5:.95]≈63.3（汇报表格片段）。

## 5. 局限性与改进方向

汇报中提到的主要局限：

- B2 合成数据的可变性不足（随机化维度有限），影响泛化收益。
- 计算资源限制导致 B2 规模受限（约 10k），且 B1 的长周期预训练占用了大量资源，挤占了生成更丰富合成数据的预算。
- 评估范围仅覆盖四个真实数据集，外部验证仍可扩展。

后续可行方向（汇报口径）：

- 增强合成数据随机化维度（光照、纹理、几何、伪影等）
- 将程序化生成与 GAN/扩散模型结合提升真实感
- 研究合成数据与真实数据的最优配比

## 6. 如何在本仓库复现

1) 安装依赖：

```bash
pip install -r requirements.txt
```

2) 准备并校验数据集（示例为 `dataset/polyps`）：

```bash
python scripts/dataset/verify_yolo_dataset.py --dataset dataset/polyps --splits train valid --num-classes 1
```

3) 训练（示例命令，按你的机器与数据规模调整 batch/device）：

```bash
python train.py --data data/polyps.yaml --img 512 512 --batch-size 16 --device 0 --name polyp-detection-yolov7
```

4) 推理可视化：

```bash
python inference/gui_detect.py --weights runs/train/polyp-detection-yolov7/weights/best.pt --device cpu
```


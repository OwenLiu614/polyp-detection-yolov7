# polyp-detection-yolov7

基于 YOLOv7 的结肠镜息肉检测项目，核心目标是验证和利用合成数据（Blender 程序化生成）来提升真实数据上的检测效果与泛化能力。

本仓库是对上游 `WongKinYiu/yolov7` 的 fork，并在此基础上补充了数据集脚本、推理脚本与项目说明。上游原始说明保留在 `docs/README_YOLOV7.md`。

项目结果与汇报要点见 `docs/RESULTS.md`（根据项目汇报文档整理）。

## Setup

```bash
pip install -r requirements.txt
```

## Repository Layout

- `data/polyps.yaml`: 默认 YOLO 数据集配置（单类 `polyp`）
- `scripts/dataset/*`: 数据集转换与校验脚本
- `inference/gui_detect.py`: 便捷图片/视频推理脚本（可选文件对话框）

## Dataset

数据集不提交到 git。默认 `data/polyps.yaml` 期望的结构：

- `dataset/polyps/images/train`
- `dataset/polyps/images/valid`
- `dataset/polyps/labels/train`
- `dataset/polyps/labels/valid`

校验数据集（检查 images/labels 对应关系与 label 格式）：

```bash
python scripts/dataset/verify_yolo_dataset.py --dataset dataset/polyps --splits train valid --num-classes 1
```

VOC XML 转 YOLO 标签（可选，用于 PolypsSet 一类数据源）：

```bash
python scripts/dataset/convert_voc_to_yolo.py --src dataset/PolypsSet --dst dataset/PolypsSet_YOLO --single-class
```

## Train

```bash
python train.py --data data/polyps.yaml --img 512 512 --batch-size 16 --device 0 --name polyp-detection-yolov7
```

## Inference

```bash
python inference/gui_detect.py --weights runs/train/polyp-detection-yolov7/weights/best.pt --device cpu
```

## Export

```bash
python export.py --weights runs/train/polyp-detection-yolov7/weights/best.pt --img-size 512 512
```

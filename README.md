# polyp-detection-yolov7

Polyp detection project built on top of YOLOv7.

This repository is a fork of the upstream YOLOv7 implementation (`WongKinYiu/yolov7`). The upstream README is kept at `docs/README_YOLOV7.md`.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Dataset files are not committed to git. Default dataset config is `data/polyps.yaml` and expects:

- `dataset/polyps/images/train`
- `dataset/polyps/images/valid`
- `dataset/polyps/labels/train`
- `dataset/polyps/labels/valid`

Verify dataset:

```bash
python scripts/dataset/verify_yolo_dataset.py --dataset dataset/polyps --splits train valid --num-classes 1
```

Convert VOC XML to YOLO labels (optional):

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

Use upstream `export.py` options as needed:

```bash
python export.py --weights runs/train/polyp-detection-yolov7/weights/best.pt --img-size 512 512
```

# polyp-detection-yolov7

This repo is a fork of YOLOv7 with a small set of project-specific utilities for a polyps dataset.

## Dataset

- Default YOLO dataset config: `data/polyps.yaml`
- Expected layout (not committed to git):
  - `dataset/polyps/images/train`
  - `dataset/polyps/images/valid`
  - `dataset/polyps/labels/train`
  - `dataset/polyps/labels/valid`

Verify dataset integrity:

```bash
python scripts/dataset/verify_yolo_dataset.py --dataset dataset/polyps --splits train valid --num-classes 1
```

Convert VOC XML to YOLO labels (example):

```bash
python scripts/dataset/convert_voc_to_yolo.py --src dataset/PolypsSet --dst dataset/PolypsSet_YOLO --single-class
```

## Inference

Image/video inference with optional file dialog:

```bash
python inference/gui_detect.py --weights runs/train/polyp-detection-yolov7/weights/best.pt --device cpu
```

Disable GUI window and save outputs:

```bash
python inference/gui_detect.py --source path/to/file.mp4 --no-view --save-dir runs/infer
```

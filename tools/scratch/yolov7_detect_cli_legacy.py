import argparse

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Legacy scratch script kept for reference. "
            "Prefer inference/gui_detect.py for day-to-day use."
        )
    )
    p.add_argument("--source", required=True, help="Path to an image or video file")
    p.add_argument(
        "--weights",
        default="runs/train/polyp-detection-yolov7/weights/best.pt",
        help="Path to .pt weights",
    )
    p.add_argument("--device", default="cpu", help="Device string, e.g. '0' or 'cpu'")
    p.add_argument("--img-size", type=int, default=512, help="Inference size (square)")
    p.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    return p.parse_args()


def preprocess(img_bgr: np.ndarray, *, img_size: int, stride: int, device: torch.device) -> torch.Tensor:
    img = letterbox(img_bgr, img_size, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndim == 3:
        img = img.unsqueeze(0)
    return img


def annotate(img_bgr: np.ndarray, det: torch.Tensor) -> np.ndarray:
    for *xyxy, conf, cls in det:
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            f"{int(cls)} {conf:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return img_bgr


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model = attempt_load(args.weights, map_location=device)
    model.eval()
    stride = int(max(model.stride)) if hasattr(model, "stride") else 32

    source = args.source
    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    is_video = source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

    def infer_on_frame(frame_bgr: np.ndarray) -> np.ndarray:
        im0 = frame_bgr.copy()
        img = preprocess(im0, img_size=args.img_size, stride=stride, device=device)
        pred = model(img)[0]
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, agnostic=False)
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            im0 = annotate(im0, det)
        return im0

    if is_image:
        im = cv2.imread(source)
        if im is None:
            raise SystemExit(f"Failed to read image: {source}")
        out = infer_on_frame(im)
        cv2.imshow("YOLOv7 Detection (Legacy)", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if is_video:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Failed to open video: {source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out = infer_on_frame(frame)
                cv2.imshow("YOLOv7 Detection (Legacy)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    raise SystemExit(f"Unsupported source type: {source}")


if __name__ == "__main__":
    main()

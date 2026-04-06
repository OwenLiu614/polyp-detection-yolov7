import argparse
from pathlib import Path
from typing import Optional


def _pick_file_dialog() -> str:
    # Lazy import so headless environments can still use --source.
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select an image or video",
        filetypes=[("Media", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")],
    )


def _is_image(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def _is_video(path: str) -> bool:
    return path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))


def run(
    *,
    weights: str,
    source: Optional[str],
    device_str: str,
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    view: bool,
    save_dir: Optional[str],
) -> None:
    # Keep imports here so `python inference/gui_detect.py --help` works without optional deps installed.
    import cv2
    import numpy as np
    import torch

    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device

    if not source:
        source = _pick_file_dialog()

    if not source:
        raise SystemExit("No source selected. Provide --source or pick a file in the dialog.")

    device = select_device(device_str)
    model = attempt_load(weights, map_location=device)
    model.eval()
    stride = int(max(model.stride)) if hasattr(model, "stride") else 32

    out_dir: Optional[Path] = Path(save_dir) if save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_bgr(frame_bgr: np.ndarray):
        im0 = frame_bgr.copy()
        im = letterbox(im0, img_size, stride=stride, auto=True)[0]
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device).float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)
        return im, im0

    def annotate(im0: np.ndarray, det: torch.Tensor) -> np.ndarray:
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                im0,
                f"{int(cls)} {conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return im0

    def infer_frame(frame_bgr: np.ndarray) -> np.ndarray:
        im, im0 = preprocess_bgr(frame_bgr)
        pred = model(im)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            im0 = annotate(im0, det)
        return im0

    if _is_image(source):
        im = cv2.imread(source)
        if im is None:
            raise SystemExit(f"Failed to read image: {source}")
        out = infer_frame(im)

        if out_dir:
            out_path = out_dir / (Path(source).stem + "_pred.jpg")
            cv2.imwrite(str(out_path), out)
            print(f"Saved: {out_path}")

        if view:
            cv2.imshow("YOLOv7 GUI Detect", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    if _is_video(source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Failed to open video: {source}")

        writer = None
        out_path = None
        if out_dir:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            out_path = out_dir / (Path(source).stem + "_pred.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            print(f"Saving video to: {out_path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out = infer_frame(frame)
                if writer:
                    writer.write(out)
                if view:
                    cv2.imshow("YOLOv7 GUI Detect", out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            if view:
                cv2.destroyAllWindows()

        if out_path:
            print(f"Saved: {out_path}")
        return

    raise SystemExit(f"Unsupported source type: {source}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple image/video inference helper for YOLOv7.")
    p.add_argument(
        "--weights",
        default="runs/train/polyp-detection-yolov7/weights/best.pt",
        help="Path to .pt weights",
    )
    p.add_argument("--source", default=None, help="Image/video path. If omitted, opens a file dialog.")
    p.add_argument("--device", default="cpu", help="Device string, e.g. '0' or 'cpu'")
    p.add_argument("--img-size", type=int, default=512, help="Inference size (square)")
    p.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--no-view", action="store_true", help="Disable cv2 window")
    p.add_argument("--save-dir", default=None, help="If set, saves outputs to this folder")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        weights=args.weights,
        source=args.source,
        device_str=args.device,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        view=not args.no_view,
        save_dir=args.save_dir,
    )

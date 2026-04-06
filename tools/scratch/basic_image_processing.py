import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Small scratch script: show basic OpenCV image transforms.")
    p.add_argument("--file", required=True, help="Path to an image file")
    p.add_argument("--save", default=None, help="If set, saves the 3-panel figure to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.file)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(blurred, cmap="gray")
    plt.title("Gaussian Blur")
    plt.axis("off")

    plt.tight_layout()
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150)
        print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()


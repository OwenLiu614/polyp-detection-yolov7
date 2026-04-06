import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def _read_label(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        # Some datasets are created with local encodings; fall back conservatively.
        return path.read_text(errors="replace").splitlines()


def _validate_label_lines(lines: List[str], *, num_classes: Optional[int]) -> List[str]:
    errors: List[str] = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {i}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except Exception:
            errors.append(f"line {i}: parse error")
            continue
        if num_classes is not None and not (0 <= cls < num_classes):
            errors.append(f"line {i}: class id {cls} out of range")
        for name, v in (("x", x), ("y", y), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                errors.append(f"line {i}: {name}={v} out of [0,1]")
                break
    return errors


def verify_split(images_dir: Path, labels_dir: Path, *, num_classes: Optional[int], max_report: int) -> int:
    images: Dict[str, Path] = {p.stem: p for p in images_dir.rglob("*") if p.is_file() and _is_image(p)}
    labels: Dict[str, Path] = {p.stem: p for p in labels_dir.rglob("*.txt") if p.is_file()}

    img_only = sorted(set(images) - set(labels))
    lbl_only = sorted(set(labels) - set(images))

    bad_labels: List[Tuple[Path, List[str]]] = []
    for stem, label_path in labels.items():
        lines = _read_label(label_path)
        errs = _validate_label_lines(lines, num_classes=num_classes)
        if errs:
            bad_labels.append((label_path, errs))

    issues = 0
    if img_only:
        issues += len(img_only)
        print(f"Missing labels for {len(img_only)} images.")
        for s in img_only[:max_report]:
            print(f"  - {images[s]}")
    if lbl_only:
        issues += len(lbl_only)
        print(f"Missing images for {len(lbl_only)} label files.")
        for s in lbl_only[:max_report]:
            print(f"  - {labels[s]}")
    if bad_labels:
        issues += len(bad_labels)
        print(f"Invalid label files: {len(bad_labels)}")
        for p, errs in bad_labels[:max_report]:
            print(f"  - {p}: {errs[0]}")

    print(f"Images: {len(images)}, Labels: {len(labels)}, Issues: {issues}")
    return issues


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify a YOLO-format dataset layout.")
    p.add_argument("--dataset", default="dataset/polyps", help="Dataset root containing images/ and labels/")
    p.add_argument("--splits", nargs="+", default=["train", "valid"], help="Splits to verify")
    p.add_argument("--num-classes", type=int, default=None, help="If set, validates class id range")
    p.add_argument("--max-report", type=int, default=20, help="Max items to print per category")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)

    total_issues = 0
    for split in args.splits:
        print(f"== Split: {split} ==")
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        if not images_dir.exists():
            print(f"Missing images dir: {images_dir}")
            total_issues += 1
            continue
        if not labels_dir.exists():
            print(f"Missing labels dir: {labels_dir}")
            total_issues += 1
            continue
        total_issues += verify_split(
            images_dir,
            labels_dir,
            num_classes=args.num_classes,
            max_report=args.max_report,
        )

    raise SystemExit(1 if total_issues else 0)


if __name__ == "__main__":
    main()


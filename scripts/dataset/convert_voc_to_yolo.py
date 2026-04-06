import argparse
import glob
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _find_image(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _iter_xml_files(annotation_dir: Path) -> Iterable[Path]:
    # Supports both:
    # - flat structure: Annotation/*.xml
    # - nested structure: Annotation/<subfolder>/*.xml
    pattern = str(annotation_dir / "**" / "*.xml")
    for p in glob.glob(pattern, recursive=True):
        yield Path(p)


def _parse_voc_objects(xml_path: Path) -> Tuple[Optional[str], List[Tuple[str, int, int, int, int]], int, int]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    filename_node = root.find("filename")
    filename = filename_node.text.strip() if filename_node is not None and filename_node.text else None

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    w = int(size.find("width").text.strip())
    h = int(size.find("height").text.strip())

    objects: List[Tuple[str, int, int, int, int]] = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bbox = obj.find("bndbox")
        if name_node is None or bbox is None:
            continue
        class_name = name_node.text.strip()
        xmin = int(bbox.find("xmin").text.strip())
        ymin = int(bbox.find("ymin").text.strip())
        xmax = int(bbox.find("xmax").text.strip())
        ymax = int(bbox.find("ymax").text.strip())
        objects.append((class_name, xmin, ymin, xmax, ymax))

    return filename, objects, w, h


def _voc_bbox_to_yolo(xmin: int, ymin: int, xmax: int, ymax: int, w: int, h: int) -> Tuple[float, float, float, float]:
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Invalid bbox (xmax<=xmin or ymax<=ymin)")
    x_center = ((xmin + xmax) / 2) / w
    y_center = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return x_center, y_center, bw, bh


def convert_split(
    *,
    src_root: Path,
    dst_root: Path,
    split: str,
    classes: List[str],
    single_class: bool,
    single_class_name: str,
    strip_filename_prefix: Optional[str],
) -> None:
    ann_dir = src_root / f"{split}2019" / "Annotation"
    img_dir = src_root / f"{split}2019" / "Image"
    if not ann_dir.exists():
        raise SystemExit(f"Annotation dir not found: {ann_dir}")
    if not img_dir.exists():
        raise SystemExit(f"Image dir not found: {img_dir}")

    out_img_dir = dst_root / "images" / split
    out_lbl_dir = dst_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    unknown_classes = set()
    missing_images = 0
    converted = 0

    for xml_path in _iter_xml_files(ann_dir):
        rel_parent = xml_path.parent.relative_to(ann_dir)
        xml_filename, objects, w, h = _parse_voc_objects(xml_path)

        # Prefer <filename> when provided; otherwise use xml stem.
        stem = Path(xml_filename).stem if xml_filename else xml_path.stem
        if strip_filename_prefix:
            stem = stem.replace(strip_filename_prefix, "")

        candidate_img_dir = img_dir / rel_parent
        image_path = _find_image(candidate_img_dir, stem) or _find_image(img_dir, stem)
        if image_path is None:
            missing_images += 1
            continue

        # Copy image
        dst_image = out_img_dir / image_path.name
        if not dst_image.exists():
            shutil.copyfile(str(image_path), str(dst_image))

        # Write label
        dst_label = out_lbl_dir / f"{Path(dst_image.name).stem}.txt"
        lines: List[str] = []
        for class_name, xmin, ymin, xmax, ymax in objects:
            if single_class:
                class_id = 0
            else:
                if class_name not in classes:
                    unknown_classes.add(class_name)
                    continue
                class_id = classes.index(class_name)

            try:
                x, y, bw, bh = _voc_bbox_to_yolo(xmin, ymin, xmax, ymax, w, h)
            except ValueError:
                continue
            lines.append(f"{class_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

        if single_class and not lines:
            # Keep empty label files consistent with typical YOLO dataset layouts.
            dst_label.write_text("")
        else:
            dst_label.write_text("".join(lines))

        converted += 1

    if single_class:
        classes_desc = f"single-class ({single_class_name})"
    else:
        classes_desc = f"{len(classes)} classes"

    print(f"[{split}] Converted: {converted}, Missing images: {missing_images}, Mode: {classes_desc}")
    if unknown_classes:
        print(f"[{split}] Unknown classes skipped ({len(unknown_classes)}): {sorted(unknown_classes)[:10]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Pascal VOC XML annotations to YOLO txt labels.")
    p.add_argument("--src", default="dataset/PolypsSet", help="VOC dataset root (contains train2019/val2019/test2019)")
    p.add_argument("--dst", default="dataset/PolypsSet_YOLO", help="Output YOLO dataset root")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to convert")
    p.add_argument("--classes", nargs="+", default=["adenomatous", "hyperplastic"], help="Class names (VOC <name>)")
    p.add_argument("--single-class", action="store_true", help="Ignore VOC class names and map everything to class 0")
    p.add_argument("--single-class-name", default="polyp", help="Only used for printing/logging when --single-class")
    p.add_argument("--strip-filename-prefix", default=None, help="Strip this prefix from <filename> stem before lookup")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        convert_split(
            src_root=src_root,
            dst_root=dst_root,
            split=split,
            classes=list(args.classes),
            single_class=bool(args.single_class),
            single_class_name=str(args.single_class_name),
            strip_filename_prefix=args.strip_filename_prefix,
        )


if __name__ == "__main__":
    main()


import os
import glob
import shutil
import xml.etree.ElementTree as ET

# 类别列表
CLASS_NAMES = ["adenomatous", "hyperplastic"]

# 数据集路径
ORIGINAL_DATASET_PATH = "D:/pythonProjects/yolov7/dataset/PolypsSet"
YOLO_DATASET_PATH = "D:/pythonProjects/yolov7/dataset/PolypsSet_YOLO"

# 确保 YOLO 目录结构存在
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "images", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "labels", split), exist_ok=True)


# 查找图片文件
def find_image(image_path, filename):
    """尝试找到匹配的图片文件（支持 .jpg, .png, .bmp）"""
    for ext in [".jpg", ".png", ".bmp"]:
        img_file = os.path.join(image_path, filename + ext)
        if os.path.exists(img_file):
            return img_file
    return None


# 转换 XML 标注为 YOLO 格式
def convert_voc_to_yolo(xml_file, txt_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    with open(txt_file, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()
            if class_name not in CLASS_NAMES:
                print(f"跳过未知类别: {class_name}")
                continue

            class_id = CLASS_NAMES.index(class_name)
            bbox = obj.find("bndbox")

            # 解析 bbox 数据并清理空格
            xmin = int(bbox.find("xmin").text.strip())
            ymin = int(bbox.find("ymin").text.strip())
            xmax = int(bbox.find("xmax").text.strip())
            ymax = int(bbox.find("ymax").text.strip())

            if xmax <= xmin or ymax <= ymin:
                print(f"无效边界框: {xml_file}")
                continue

            # 转换为 YOLO 格式
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# 处理训练集（没有子文件夹）
def process_train_dataset():
    annotation_path = os.path.join(ORIGINAL_DATASET_PATH, "train2019", "Annotation")
    image_path = os.path.join(ORIGINAL_DATASET_PATH, "train2019", "Image")

    for xml_file in glob.glob(os.path.join(annotation_path, "*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 直接从 XML 读取 `<filename>`
        img_filename = root.find("filename").text.strip()
        img_filename_without_ext = os.path.splitext(img_filename)[0]  # 去掉扩展名

        # **去掉 `adenoma_` 这样的前缀**
        img_filename_without_ext = img_filename_without_ext.replace("adenoma_", "")

        img_file = find_image(image_path, img_filename_without_ext)
        if img_file is None:
            print(f"⚠️ 训练集: 图片文件不存在 {os.path.join(image_path, img_filename_without_ext)}，跳过...")
            continue

        # 读取图片尺寸
        size = root.find("size")
        img_width = int(size.find("width").text.strip())
        img_height = int(size.find("height").text.strip())

        # 目标路径
        new_img_path = os.path.join(YOLO_DATASET_PATH, "images", "train", os.path.basename(img_file))
        new_txt_path = os.path.join(YOLO_DATASET_PATH, "labels", "train", img_filename_without_ext + ".txt")

        # 复制图片并转换 XML 标注
        shutil.copy(img_file, new_img_path)
        convert_voc_to_yolo(xml_file, new_txt_path, img_width, img_height)


# 处理验证集（val）和测试集（test） - **有子文件夹**
def process_other_datasets(split):
    annotation_path = os.path.join(ORIGINAL_DATASET_PATH, f"{split}2019", "Annotation")
    image_path = os.path.join(ORIGINAL_DATASET_PATH, f"{split}2019", "Image")

    for subfolder in os.listdir(annotation_path):
        annotation_subfolder = os.path.join(annotation_path, subfolder)
        image_subfolder = os.path.join(image_path, subfolder)

        for xml_file in glob.glob(os.path.join(annotation_subfolder, "*.xml")):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 直接从 XML 读取 `<filename>`
            img_filename = root.find("filename").text.strip()
            img_filename_without_ext = os.path.splitext(img_filename)[0]  # 去掉扩展名

            # **去掉 `adenoma_` 这样的前缀**
            img_filename_without_ext = img_filename_without_ext.replace("adenoma_", "")

            img_file = find_image(image_subfolder, img_filename_without_ext)
            if img_file is None:
                print(
                    f"⚠️ {split} 集: 图片文件不存在 {os.path.join(image_subfolder, img_filename_without_ext)}，跳过...")
                continue

            # 读取图片尺寸
            size = root.find("size")
            img_width = int(size.find("width").text.strip())
            img_height = int(size.find("height").text.strip())

            # 目标路径
            new_img_path = os.path.join(YOLO_DATASET_PATH, "images", split, os.path.basename(img_file))
            new_txt_path = os.path.join(YOLO_DATASET_PATH, "labels", split, img_filename_without_ext + ".txt")

            # 复制图片并转换 XML 标注
            shutil.copy(img_file, new_img_path)
            convert_voc_to_yolo(xml_file, new_txt_path, img_width, img_height)


# 运行转换
process_train_dataset()  # 处理训练集
process_other_datasets("val")  # 处理验证集
process_other_datasets("test")  # 处理测试集
print("✅ 数据集转换完成！")

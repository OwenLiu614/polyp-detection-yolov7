import os

image_dir = "D:/pythonProjects/yolov7/dataset/polyps/images/train"
missing_images = []

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        missing_images.append(image_path)

if missing_images:
    print(f"❌ 发现 {len(missing_images)} 张图片缺失！")
    print(missing_images[:10])  # 只显示前 10 个
else:
    print("✅ 所有图片都存在！")

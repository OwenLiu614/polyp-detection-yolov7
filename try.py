import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device

# 选择文件
root = tk.Tk()
root.withdraw()  # 隐藏主窗口
file_path = filedialog.askopenfilename(title="选择图片或视频",
                                       filetypes=[("图片/视频文件", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")])

if not file_path:
    print("❌ 未选择文件，程序退出！")
    exit()

# 加载 YOLOv7 模型
model_path = "runs/train/yolov7-polyps/weights/best.pt"
device = select_device("0")  # 使用 GPU
model = attempt_load(model_path, map_location=device)
model.eval()

# 设置输入大小
img_size = 512

# 颜色
colors = (0, 255, 0)  # 绿色


# 处理图片
def detect_image(img):
    img0 = img.copy()
    img = letterbox(img, img_size, stride=32, auto=True)[0]  # 预处理
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, CHW
    img = np.ascontiguousarray(img) / 255.0  # 归一化

    # 转换为 tensor
    img = torch.from_numpy(img).to(device)
    img = img.float().unsqueeze(0)  # (1, 3, 512, 512)

    # 推理
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), colors, 2)
                label = f"Polyp {conf:.2f}"
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors, 2)

    return img0


# 处理视频
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_image(frame)
        cv2.imshow("YOLOv7 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 按 Q 退出
            break
    cap.release()
    cv2.destroyAllWindows()


# 运行检测
if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
    img = cv2.imread(file_path)
    result = detect_image(img)
    cv2.imshow("YOLOv7 Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif file_path.lower().endswith((".mp4", ".avi", ".mov")):
    detect_video(file_path)
else:
    print("❌ 不支持的文件格式！")

FROM ubuntu:latest
LABEL authors="Owenliu"

ENTRYPOINT ["top", "-b"]

# 使用官方 PyTorch CUDA 版本
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# 设置工作目录
WORKDIR /workspace

# 复制 Python 代码
COPY *.py /workspace/
COPY utils /workspace/utils/

# 复制模型文件（.pt）
COPY *.pt /workspace/

# 复制 `requirements.txt`
COPY requirements.txt /workspace/

# 更新系统并安装 OpenCV 相关依赖（解决 libGL.so.1 问题）
RUN apt update && apt install -y \
    git wget vim \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*  # 清理缓存减少镜像大小

# 更新 pip 并安装 Python 依赖
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 进入 bash 交互模式
CMD ["bash"]

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
LABEL authors="Owenliu"

WORKDIR /workspace

COPY requirements.txt /workspace/
COPY *.py /workspace/
COPY models /workspace/models/
COPY utils /workspace/utils/
COPY inference /workspace/inference/
COPY scripts /workspace/scripts/

# OpenCV runtime deps + basic tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget vim \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD ["bash"]


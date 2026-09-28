FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    -r requirements.txt

COPY app.py .
COPY resnet50_cifar10_best.pth .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
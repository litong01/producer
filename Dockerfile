FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only first to avoid pulling CUDA variants
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchcodec

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Demucs model so it's baked into the image
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

COPY app/ app/

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1", "--threads", "4", "app.main:create_app()"]

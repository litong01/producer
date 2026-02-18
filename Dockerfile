FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-eng \
    poppler-utils \
    libheif-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# All pip installs + model downloads in one layer so cleanup saves space.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps oemer \
    && python -c "from demucs.pretrained import get_model; get_model('htdemucs')" \
    && python -c "import oemer; print('oemer OK')" \
    && SITE=/usr/local/lib/python3.11/site-packages \
    && rm -rf \
        $SITE/torch/test \
        $SITE/torch/benchmarks \
        $SITE/torch/utils/benchmark \
        $SITE/torch/utils/bottleneck \
        $SITE/torch/utils/viz \
        $SITE/torch/include \
        $SITE/torch/share \
        $SITE/torch/lib/*.a \
        $SITE/torchaudio/lib/*.a \
        $SITE/nvidia \
        $SITE/caffe2 \
    && find $SITE -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null \
    && find $SITE -name '*.pyc' -delete 2>/dev/null \
    && rm -rf /root/.cache /tmp/* \
    ; true

COPY app/ app/

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1", "--threads", "4", "app.main:create_app()"]

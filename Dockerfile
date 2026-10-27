# CPU-only reproduction container (the reported runs used a local RTX 5050 GPU).
# GPU use: install a CUDA-enabled torch wheel (see README) and run with `--gpus all`.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data preparation only; training on CPU is possible but very slow.
CMD ["python", "scripts/00_prepare_data.py"]

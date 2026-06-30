# ── Build stage ───────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

RUN apt-get update -qq && apt-get install -y -qq \
    python3.11 python3.11-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY server/requirements.txt .
RUN python3.11 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir \
        torch==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    /venv/bin/pip install --no-cache-dir \
        -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Pre-download model weights into image layer
# (Alternatively, mount a volume and set HF_HOME to a persistent cache path)
ARG HF_MODEL=Mavkif/m2m100_rup_ur_to_rur
ARG HF_TOKENIZER=Mavkif/m2m100_rup_tokenizer_both
ENV HF_HOME=/root/.cache/huggingface
RUN /venv/bin/python -c " \
    import os; \
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer; \
    model_id = os.environ.get('HF_MODEL', 'Mavkif/m2m100_rup_ur_to_rur'); \
    tokenizer_id = os.environ.get('HF_TOKENIZER', 'Mavkif/m2m100_rup_tokenizer_both'); \
    M2M100Tokenizer.from_pretrained(tokenizer_id); \
    M2M100ForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True) \
    "

COPY server/app.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["/venv/bin/uvicorn", "app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--loop", "uvloop", "--log-level", "info"]

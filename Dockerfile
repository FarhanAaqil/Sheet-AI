# ---------- Stage 1: Build ----------
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy application source
COPY agent.py main.py app.py ./

RUN mkdir -p /app/cache

# Expose both FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default: run FastAPI. Override with CMD ["streamlit","run","app.py",...] for UI.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

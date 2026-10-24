# ---------- Stage 1: Frontend build ----------
FROM node:22-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# ---------- Stage 2: Python dependencies ----------
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 3: Runtime ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SQLITE_DB_PATH=/app/data/sheetsense.db

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy application source and the canonical web client
COPY *.py ./
COPY docs ./docs
COPY alembic.ini ./
COPY migrations ./migrations
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN addgroup --system sheetsense \
    && adduser --system --ingroup sheetsense sheetsense \
    && mkdir -p /app/cache /app/data \
    && chown -R sheetsense:sheetsense /app/cache /app/data

USER sheetsense


# Expose the FastAPI gateway and bundled browser client
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]

# Default: run FastAPI; the React client is served from /.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

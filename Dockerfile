# =============================================================================
# AFDGCN — Phase API Production Docker Image
# =============================================================================

FROM python:3.11-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Önce sadece requirements — layer cache avantajı
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodu
COPY . .

# Port
EXPOSE 9001

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:9001/health || exit 1

# Phase API başlat (tek worker — AFDGCN thread-safe değil)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "9001", "--workers", "1"]

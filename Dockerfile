# =============================================================================
# AFDGCN — Python Model Server (port 9002)
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
COPY requirements_model.txt .
RUN pip install --no-cache-dir -r requirements_model.txt

# Uygulama kodu
COPY model_server.py config.py ./
COPY model/ ./model/
COPY ml/ ./ml/
COPY conf/ ./conf/
COPY data/Kayseri/ ./data/Kayseri/
COPY saved_models/ ./saved_models/

# config.py Kayseri_AFDGCN.conf arar; depoda Kayseri_Serit_AFDGCN.conf var.
RUN cp ./conf/Kayseri_Serit_AFDGCN.conf ./conf/Kayseri_AFDGCN.conf

# Port
EXPOSE 9002

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9002/health || exit 1

# Model sunucusunu başlat (tek worker — AFDGCN thread-safe değil)
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "9002", "--workers", "1"]

# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile — Movie Matrix AI Pro
#
# Kullanım (local):
#   docker build -t movie-matrix .
#   docker run -p 8501:8501 movie-matrix
#
# Not: Google Colab'da Docker çalışmaz.
#      Colab için movie_agent_colab.ipynb dosyasını kullan.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları (önce kopyala → cache optimizasyonu)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyaları
COPY . .

# Streamlit portu
EXPOSE 8501

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Başlatma komutu
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

FROM python:3.10-slim

WORKDIR /app

# system build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

# install torch CPU-only first (needs a separate index URL)
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

# install torch-geometric and remaining deps
RUN pip install --no-cache-dir torch-geometric==2.5.3

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# app source
COPY src/ src/

# pre-built graph artifacts and trained model (~22 MB total)
COPY data/graph/ data/graph/
COPY model_best.pt .

ENV PYTHONPATH=/app
ENV API_BASE=http://localhost:8000

EXPOSE 7860

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]

#!/bin/bash
set -e

# Start the FastAPI backend in the background
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Wait for the API to be ready before starting Streamlit
echo "Waiting for API to start..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
  sleep 1
done
echo "API ready."

# Start the Streamlit frontend on port 7860 (Hugging Face Spaces default)
exec streamlit run src/frontend/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  --server.enableCORS false \
  --server.enableXsrfProtection false

# Regression (MMM) — Streamlit serving image.  # -- scaffolding --
# Authored + documented per program decision: image is NOT built locally (no Docker
# Desktop). Build/verify on any Docker host:
#   docker build -t regression-mmm .
#   docker run -p 8501:8501 regression-mmm
# No torch, no xgboost/lightgbm — pure numpy/pandas/scikit-learn/scipy/statsmodels, so no
# libomp/OpenMP juggling and no CUDA base needed.
FROM python:3.11-slim

WORKDIR /app

# libgomp1: OpenMP runtime that sklearn/statsmodels wheels link on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py app.py ./
COPY src/ src/
# fitted artifacts must exist already (run run_pipeline.py + evaluate.py first)
COPY outputs/models/ outputs/models/
COPY outputs/metrics.json outputs/decomposition.json outputs/budget_reallocation.json \
     outputs/diagnostics.json outputs/pipeline_params.json outputs/
COPY data/processed/mmm_panel.parquet data/processed/mmm_panel.parquet

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true", \
     "--server.address", "0.0.0.0", "--server.port", "8501"]

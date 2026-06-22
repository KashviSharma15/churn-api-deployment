# ── Stage 1: base image ──────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file first (layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY app/         ./app/
COPY model.pkl    .
COPY scaler.pkl   .
COPY feature_names.pkl .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

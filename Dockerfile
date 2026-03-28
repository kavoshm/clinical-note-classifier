# =============================================================================
# Clinical Note Classifier — Dockerfile
# =============================================================================
# Runs the Gradio web interface for classifying clinical notes into structured
# outputs: urgency level, ICD-10 coding, clinical reasoning, and recommended
# actions. Supports both demo mode (pre-computed results) and live mode
# (requires OPENAI_API_KEY).
#
# Build:  docker build -t clinical-note-classifier .
# Run:    docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... clinical-note-classifier
# =============================================================================

# --- Stage 1: Base image ---
# Use Python 3.11 slim for a smaller image footprint while retaining
# all necessary system libraries for the scientific Python stack.
FROM python:3.11-slim AS base

# --- Stage 2: Set working directory ---
# All subsequent commands run relative to /app inside the container.
WORKDIR /app

# --- Stage 3: Install system dependencies ---
# Some Python packages (numpy, pandas, matplotlib) require build tools.
# Install them in a single layer, then clean up to reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 4: Install Python dependencies ---
# Copy requirements.txt first to leverage Docker layer caching.
# Dependencies change less frequently than source code, so this layer
# is cached across builds when only source code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 5: Copy application source code ---
# Copy the full application after dependencies are installed.
COPY src/ src/
COPY data/ data/
COPY outputs/ outputs/
COPY app.py .

# --- Stage 6: Create non-root user for security ---
# Running as root inside a container is a security risk. Create a
# dedicated user with no login shell and switch to it.
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# --- Stage 7: Expose the Gradio default port ---
# Gradio serves on port 7860 by default.
EXPOSE 7860

# --- Stage 8: Configure environment ---
# OPENAI_API_KEY is passed at runtime via -e flag or docker-compose.
# Gradio needs server_name="0.0.0.0" to be accessible outside the container.
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV PYTHONUNBUFFERED=1

# --- Stage 9: Launch the Gradio app ---
CMD ["python", "app.py"]

# Dockerfile for AI Music Genre Classification Web App
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Install system audio libraries (libsndfile & ffmpeg for audio format decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list and install Python wheels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, templates, and pre-trained weights
COPY app.py .
COPY predict_genre.py .
COPY templates/ ./templates/
COPY static/ ./static/
COPY gtzan_project/ ./gtzan_project/
COPY results_gtzan_high_acc/ ./results_gtzan_high_acc/

# Hugging Face Spaces uses port 7860; default Render/Docker uses 5000
EXPOSE 5000 7860

# Run Flask web server
CMD ["python", "app.py"]

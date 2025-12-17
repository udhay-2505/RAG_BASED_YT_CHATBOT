# Use Python 3.9
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install ffmpeg (useful for yt-dlp) and git
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Create a directory for transcripts and give permission to write
RUN mkdir -p /app/transcripts
RUN chmod 777 /app/transcripts

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend_api.py .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the app
# Host must be 0.0.0.0 to be accessible externally
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "7860"]
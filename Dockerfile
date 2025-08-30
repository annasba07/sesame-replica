# Hybrid Therapy AI - Docker Configuration
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_hybrid.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_hybrid.txt

# Copy application files
COPY hybrid_therapy_system.py .
COPY web_interface.py .
COPY config.json .

# Create directories for models and logs
RUN mkdir -p checkpoints logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["python", "web_interface.py"]
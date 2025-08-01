# Shakespeare Chatbot - Docker Container
# Multi-stage build for efficient container size

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG TRAINING_ITERS=5000
ARG MODEL_SIZE=small

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/shakespeare models/shakespeare_model static templates

# Download and prepare Shakespeare data
RUN python src/prepare_data.py

# Train the model during build (this takes time but creates a ready-to-use container)
RUN echo "Training model with $TRAINING_ITERS iterations..." && \
    python src/train.py --max_iters $TRAINING_ITERS

# Production stage
FROM python:3.9-slim as production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files from builder stage
COPY --from=builder /app/src ./src
COPY --from=builder /app/config ./config
COPY --from=builder /app/static ./static
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/data ./data
COPY --from=builder /app/models ./models

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Create volume mount points for persistence
VOLUME ["/app/models", "/app/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/shakespeare_model
ENV DATA_PATH=/app/data/shakespeare
ENV PORT=8080
ENV HOST=0.0.0.0

# Default command
CMD ["python", "src/chat_interface.py"]

# Labels for metadata
LABEL maintainer="your-email@example.com"
LABEL description="Shakespeare Chatbot using nanoGPT"
LABEL version="1.0"
LABEL org.opencontainers.image.source="https://github.com/your-username/shakespeare-chatbot"
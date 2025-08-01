# Shakespeare Chatbot - nanoGPT Project Makefile
# Provides automation for setup, training, and running the chatbot

.PHONY: help setup prepare-data train train-cpu chat sample clean docker-build docker-run docker-stop test

# Default target
help:
	@echo "Shakespeare Chatbot - Available Commands:"
	@echo ""
	@echo "Setup & Data:"
	@echo "  setup         - Create virtual environment and install dependencies"
	@echo "  prepare-data  - Download and prepare Shakespeare dataset"
	@echo ""
	@echo "Training:"
	@echo "  train         - Train the nanoGPT model (GPU/CPU auto-detect)"
	@echo "  train-cpu     - Force CPU-only training (slower but universal)"
	@echo ""
	@echo "Running:"
	@echo "  chat          - Start the web-based chat interface"
	@echo "  sample        - Generate sample text from trained model"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  - Build Docker container (includes training)"
	@echo "  docker-run    - Run complete application in Docker"
	@echo "  docker-stop   - Stop Docker containers"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         - Clean up generated files and cache"
	@echo "  test          - Run basic functionality tests"
	@echo ""

# Variables
VENV_NAME := venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip
DATA_DIR := data/shakespeare
MODEL_DIR := models/shakespeare_model

# Setup virtual environment and dependencies
setup:
	@echo "🔧 Setting up virtual environment..."
	python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Setup complete! Virtual environment created with all dependencies."

# Download and prepare Shakespeare dataset
prepare-data:
	@echo "📚 Preparing Shakespeare dataset..."
	mkdir -p $(DATA_DIR)
	$(PYTHON) src/prepare_data.py
	@echo "✅ Shakespeare dataset prepared in $(DATA_DIR)/"

# Train the model (auto-detect GPU/CPU)
train: prepare-data
	@echo "🚂 Starting model training..."
	mkdir -p $(MODEL_DIR)
	$(PYTHON) src/train.py --config config/train_config.py
	@echo "✅ Training complete! Model saved in $(MODEL_DIR)/"

# Force CPU-only training
train-cpu: prepare-data
	@echo "🚂 Starting CPU-only model training (this may take 2-3 hours)..."
	mkdir -p $(MODEL_DIR)
	$(PYTHON) src/train.py --config config/train_config.py --device cpu --compile False
	@echo "✅ CPU training complete! Model saved in $(MODEL_DIR)/"

# Start the web chat interface
chat:
	@echo "💬 Starting chat interface..."
	@echo "🌐 Open your browser to http://localhost:8080"
	$(PYTHON) src/chat_interface.py

# Generate sample text
sample:
	@echo "📝 Generating sample Shakespeare text..."
	$(PYTHON) src/model_utils.py --mode sample
	@echo "✅ Sample generation complete!"

# Test basic functionality
test:
	@echo "🧪 Running tests..."
	@echo "Testing data preparation..."
	@test -f $(DATA_DIR)/train.bin || (echo "❌ Training data not found" && exit 1)
	@test -f $(DATA_DIR)/val.bin || (echo "❌ Validation data not found" && exit 1)
	@echo "Testing model files..."
	@test -f $(MODEL_DIR)/ckpt.pt || (echo "❌ Model checkpoint not found - run 'make train' first" && exit 1)
	@echo "Testing model loading..."
	$(PYTHON) src/model_utils.py --mode test
	@echo "✅ All tests passed!"

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf $(DATA_DIR)/*.bin $(DATA_DIR)/*.pkl
	rm -rf $(MODEL_DIR)/*
	rm -rf __pycache__ src/__pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Cleanup complete!"

# Docker commands
docker-build:
	@echo "🐳 Building Docker container..."
	docker build -t shakespeare-chatbot .
	@echo "✅ Docker container built successfully!"

docker-run:
	@echo "🐳 Running Docker container..."
	@echo "🌐 Open your browser to http://localhost:8080"
	docker run -d --name shakespeare-chatbot-container \
		-p 8080:8080 \
		-v $(shell pwd)/models:/app/models \
		shakespeare-chatbot
	@echo "✅ Container is running! Use 'make docker-stop' to stop."

docker-stop:
	@echo "🐳 Stopping Docker container..."
	-docker stop shakespeare-chatbot-container
	-docker rm shakespeare-chatbot-container
	@echo "✅ Container stopped."

# Development helpers
dev-server: chat

install-dev: setup
	$(PIP) install pytest black flake8 jupyter
	@echo "✅ Development dependencies installed!"

format:
	$(VENV_NAME)/bin/black src/
	@echo "✅ Code formatted with black!"

lint:
	$(VENV_NAME)/bin/flake8 src/
	@echo "✅ Code linted!"

# Quick start for new users
quickstart: setup prepare-data train chat

# Check if virtual environment exists
check-venv:
	@test -d $(VENV_NAME) || (echo "❌ Virtual environment not found. Run 'make setup' first." && exit 1)

# Add dependency to most targets
prepare-data train train-cpu chat sample test: check-venv

# Training variants
train-small: prepare-data
	@echo "🚂 Training small model (faster, less capable)..."
	mkdir -p $(MODEL_DIR)
	$(PYTHON) src/train.py --config config/train_config.py \
		--n_layer 4 --n_head 4 --n_embd 256 --max_iters 2000
	@echo "✅ Small model training complete!"

train-large: prepare-data
	@echo "🚂 Training large model (slower, more capable)..."
	mkdir -p $(MODEL_DIR)
	$(PYTHON) src/train.py --config config/train_config.py \
		--n_layer 8 --n_head 8 --n_embd 512 --max_iters 10000
	@echo "✅ Large model training complete!"

# Monitoring
monitor-training:
	@echo "📊 Monitoring training progress..."
	tail -f $(MODEL_DIR)/training.log

# Backup model
backup-model:
	@echo "💾 Backing up trained model..."
	tar -czf shakespeare_model_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz $(MODEL_DIR)/
	@echo "✅ Model backed up!"

# Show project status
status:
	@echo "📊 Project Status:"
	@echo ""
	@echo "Virtual Environment:"
	@if [ -d $(VENV_NAME) ]; then echo "  ✅ Created"; else echo "  ❌ Not found - run 'make setup'"; fi
	@echo ""
	@echo "Data:"
	@if [ -f $(DATA_DIR)/train.bin ]; then echo "  ✅ Training data ready"; else echo "  ❌ Not prepared - run 'make prepare-data'"; fi
	@if [ -f $(DATA_DIR)/val.bin ]; then echo "  ✅ Validation data ready"; else echo "  ❌ Not prepared - run 'make prepare-data'"; fi
	@echo ""
	@echo "Model:"
	@if [ -f $(MODEL_DIR)/ckpt.pt ]; then echo "  ✅ Model trained and ready"; else echo "  ❌ Not trained - run 'make train'"; fi
	@echo ""
	@echo "Docker:"
	@if docker images | grep -q shakespeare-chatbot; then echo "  ✅ Docker image built"; else echo "  ❌ Not built - run 'make docker-build'"; fi
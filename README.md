# shakespeare-chatbot

# Shakespeare Chatbot - nanoGPT Project

A complete implementation of a Shakespeare-style chatbot using nanoGPT, trained on Shakespeare's complete works. This project includes Docker containerization, a web-based chat interface, and automated setup via Makefile.

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)
```bash
# Clone and setup
git clone https://github.com/kumogire/shakespeare-chatbot.git
cd shakespeare-chatbot

# Build and run with Docker
make docker-build
make docker-run

# Access the chat interface at http://localhost:8080
```

### Option 2: Local Development
```bash
# Setup virtual environment and dependencies
make setup

# Download and prepare data
make prepare-data

# Train the model (takes ~30 min with GPU, 2-3 hours with CPU)
make train

# Start the chat interface
make chat
```

## 📁 Project Structure

```
shakespeare-chatbot/
├── README.md                  # This file
├── Makefile                   # Automation commands
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Multi-service orchestration
├── requirements.txt           # Python dependencies
├── src/
│   ├── train.py              # Model training script
│   ├── chat_interface.py     # Web-based chat interface
│   ├── model_utils.py        # Model loading/generation utilities
│   └── prepare_data.py       # Data preparation script
├── config/
│   └── train_config.py       # Training configuration
├── data/
│   └── shakespeare/          # Training data (auto-downloaded)
├── models/
│   └── shakespeare_model/    # Saved model checkpoints
├── static/
│   ├── style.css            # Web interface styling
│   └── script.js            # Frontend JavaScript
└── templates/
    └── chat.html            # Chat interface template
```

## 🛠️ Prerequisites

### For Local Development:
- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster training

### For Docker:
- Docker
- Docker Compose (optional)

## 📋 Available Make Commands

| Command | Description |
|---------|-------------|
| `make setup` | Create virtual environment and install dependencies |
| `make prepare-data` | Download and prepare Shakespeare dataset |
| `make train` | Train the nanoGPT model on Shakespeare |
| `make train-cpu` | Train using CPU only (slower but works everywhere) |
| `make chat` | Start the web-based chat interface |
| `make sample` | Generate sample text from trained model |
| `make clean` | Clean up generated files and cache |
| `make docker-build` | Build Docker container |
| `make docker-run` | Run the complete application in Docker |
| `make docker-stop` | Stop Docker containers |
| `make test` | Run basic functionality tests |

## 🏃‍♂️ Detailed Setup Instructions

### Step 1: Clone and Enter Project
```bash
git clone https://github.com/kumogire/shakespeare-chatbot.git
cd shakespeare-chatbot
```

### Step 2: Choose Your Setup Method

#### Method A: Docker (Easiest)
```bash
# Build the container (includes model training)
make docker-build

# Run the application
make docker-run

# Open your browser to http://localhost:8080
```

The Docker build will automatically:
- Set up the Python environment
- Download Shakespeare's works
- Train the model
- Start the web interface

#### Method B: Local Development
```bash
# 1. Setup environment
make setup

# 2. Prepare training data
make prepare-data

# 3. Train the model
make train  # or 'make train-cpu' for CPU-only

# 4. Start chat interface
make chat
```

### Step 3: Using the Chat Interface

1. Open your browser to `http://localhost:8080`
2. Type messages in Shakespearean style
3. See the AI respond in Shakespeare's voice!

Example conversation:
```
You: "How fares thee this day?"
Bot: "Fair sir, mine spirits are lifted high, as doth the lark at break of day sing sweetest melodies..."
```

## ⚙️ Configuration

### Training Parameters
Edit `config/train_config.py` to customize:

```python
# Model size
n_layer = 6        # Number of transformer layers
n_head = 6         # Number of attention heads  
n_embd = 384       # Embedding dimension

# Training
max_iters = 5000   # Training iterations
batch_size = 64    # Batch size
learning_rate = 1e-3

# Hardware
device = 'cuda'    # 'cuda' for GPU, 'cpu' for CPU only
```

### Web Interface
- **Port**: Default 8080 (change in `src/chat_interface.py`)
- **Styling**: Modify `static/style.css`
- **Behavior**: Edit `static/script.js`

## 🧪 Testing Your Setup

```bash
# Test data preparation
make test-data

# Test model loading
make test-model

# Generate sample text
make sample
```

## 📊 Training Details

### Expected Training Times:
- **GPU (RTX 3080)**: ~20-30 minutes
- **GPU (GTX 1060)**: ~45-60 minutes  
- **CPU (Modern i7)**: ~2-3 hours
- **CPU (Older hardware)**: ~4-6 hours

### Training Progress:
Watch for decreasing loss values:
```
step 0: train loss 4.278, val loss 4.277     # Random text
step 1000: train loss 1.892, val loss 1.945  # Learning words
step 3000: train loss 1.234, val loss 1.456  # Learning grammar  
step 5000: train loss 0.823, val loss 1.234  # Coherent Shakespeare!
```

### Model Size:
- **Parameters**: ~10M (much smaller than GPT-3's 175B!)
- **File Size**: ~40MB
- **RAM Usage**: ~500MB during inference

## 🐳 Docker Details

### Container Features:
- **Base**: Python 3.9 slim
- **Auto-training**: Model trains during container build
- **Volume mounting**: Persist models between container runs
- **Port mapping**: 8080 (host) → 8080 (container)

### Docker Commands:
```bash
# Build with custom config
docker build --build-arg TRAINING_ITERS=3000 .

# Run with different port
docker run -p 5000:8080 shakespeare-chatbot

# Mount local model directory
docker run -v $(pwd)/models:/app/models shakespeare-chatbot
```

## 🔧 Troubleshooting

### Common Issues:

**"CUDA out of memory"**
```bash
# Use CPU training instead
make train-cpu
```

**"Port 8080 already in use"**
```bash
# Change port in src/chat_interface.py or use different port
docker run -p 5000:8080 shakespeare-chatbot
```

**"Model not found"**
```bash
# Ensure training completed successfully
make train
# Check for model files
ls models/shakespeare_model/
```

**Slow training on CPU**
```bash
# Reduce model size in config/train_config.py
n_layer = 4      # instead of 6
max_iters = 2000 # instead of 5000
```

## 🎯 Next Steps

### Experiment with:
1. **Different datasets**: Replace Shakespeare with your own text
2. **Model architecture**: Adjust layers, attention heads, embedding size
3. **Training duration**: Longer training = better quality
4. **Interface improvements**: Add conversation history, user profiles
5. **Deployment**: Deploy to cloud platforms (AWS, Google Cloud, etc.)

### Advanced Features to Add:
- Conversation memory/context
- Multiple character personalities  
- Fine-tuning on specific plays
- REST API endpoints
- Real-time training updates

## 📄 License

MIT License - Feel free to use this project for learning and experimentation!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📚 Resources

- [nanoGPT Original Repository](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need Paper](https://arxiv.org/abs/1706.03762)
- [GPT Architecture Explained](https://jalammar.github.io/illustrated-gpt2/)
- [Shakespeare Complete Works](http://www.gutenberg.org/ebooks/100)
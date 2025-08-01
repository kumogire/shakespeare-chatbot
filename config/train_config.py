"""
Training Configuration for Shakespeare nanoGPT

This file contains all the hyperparameters and settings for training
the Shakespeare model. Modify these values to experiment with different
model architectures and training strategies.
"""

import torch

# ==============================================================================
# I/O CONFIGURATION
# ==============================================================================

# Output directory for saving model checkpoints and logs
out_dir = 'models/shakespeare_model'

# How often to evaluate the model and print stats
eval_interval = 250

# How often to print training stats
log_interval = 10

# Number of batches to use for evaluation
eval_iters = 200

# If True, only run evaluation and exit (useful for testing)
eval_only = False

# If True, always save a checkpoint after each evaluation
# If False, only save when validation loss improves
always_save_checkpoint = True

# Initialization mode:
# 'scratch' - initialize a new model from scratch
# 'resume' - resume training from a saved checkpoint
# 'gpt2' - initialize from a pre-trained GPT-2 model (not implemented)
init_from = 'scratch'

# ==============================================================================
# WANDB LOGGING (Optional)
# ==============================================================================

# Enable Weights & Biases logging for experiment tracking
wandb_log = False

# W&B project name
wandb_project = 'shakespeare-nanogpt'

# W&B run name (will be auto-generated if None)
wandb_run_name = None

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

# Dataset name (used for loading the correct data files)
dataset = 'shakespeare'

# Gradient accumulation steps - simulates larger batch sizes
# Total effective batch size = batch_size * gradient_accumulation_steps
gradient_accumulation_steps = 1

# Batch size per GPU
batch_size = 64

# Context length - maximum sequence length the model can handle
# Shakespeare works well with 256, but you can try 512 or 1024 for longer context
block_size = 256

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

# Number of transformer layers
# More layers = more powerful but slower training
# 6 layers is good for Shakespeare, try 8-12 for more complex tasks
n_layer = 6

# Number of attention heads per layer
# Should divide n_embd evenly
# 6 heads works well with 384 embedding dim
n_head = 6

# Embedding dimension
# Larger = more expressive but more parameters
# 384 is good for Shakespeare (10M parameters)
# Try 512 or 768 for larger models
n_embd = 384

# Dropout rate for regularization
# 0.0 for pre-training, 0.1-0.2 for fine-tuning
# Higher values prevent overfitting but may hurt performance
dropout = 0.2

# Whether to use bias in Linear and LayerNorm layers  
# False is slightly better and faster (following recent trends)
bias = False

# ==============================================================================
# OPTIMIZER CONFIGURATION (AdamW)
# ==============================================================================

# Maximum learning rate
# 1e-3 works well for small models like this
# Larger models often need smaller LR (1e-4 to 6e-4)
learning_rate = 1e-3

# Total number of training iterations
# 5000 is good for Shakespeare, increase for larger datasets
max_iters = 5000

# L2 regularization weight decay
weight_decay = 1e-1

# Adam beta parameters
beta1 = 0.9
beta2 = 0.95

# Gradient clipping value (0.0 to disable)
# Helps with training stability
grad_clip = 1.0

# ==============================================================================
# LEARNING RATE SCHEDULE
# ==============================================================================

# Whether to use learning rate decay
decay_lr = True

# Number of warmup steps (gradual LR increase at start)
warmup_iters = 100

# Total steps for LR decay (usually equal to max_iters)
lr_decay_iters = 5000

# Minimum learning rate (usually learning_rate / 10)
min_lr = 1e-4

# ==============================================================================
# SYSTEM CONFIGURATION
# ==============================================================================

# Device selection
# 'auto' - automatically choose GPU if available, else CPU
# 'cpu' - force CPU usage
# 'cuda' - use default GPU
# 'cuda:0', 'cuda:1' - use specific GPU
# 'mps' - use Apple Metal (Mac M1/M2)
device = 'auto'

# Data type for training
# 'float32' - highest precision, slowest
# 'float16' - mixed precision, faster on modern GPUs  
# 'bfloat16' - better stability than float16, requires newer GPUs
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# Whether to compile the model with PyTorch 2.0
# Significant speedup on modern GPUs, but requires PyTorch >= 2.0
compile = True

# ==============================================================================
# DISTRIBUTED TRAINING (Multi-GPU)
# ==============================================================================

# Backend for multi-GPU training
# 'nccl' for NVIDIA GPUs, 'gloo' for CPU or mixed setups
backend = 'nccl'

# ==============================================================================
# PRESETS FOR DIFFERENT SCENARIOS
# ==============================================================================

# Uncomment one of these sections to use predefined configurations:

# # FAST TRAINING (for testing/debugging)
# max_iters = 1000
# eval_interval = 100
# batch_size = 32
# n_layer = 4
# n_head = 4
# n_embd = 256

# # SMALL MODEL (for limited compute)
# n_layer = 4
# n_head = 4  
# n_embd = 256
# batch_size = 32
# max_iters = 3000

# # LARGE MODEL (for better quality, requires more compute)
# n_layer = 8
# n_head = 8
# n_embd = 512
# batch_size = 32  # Reduce if out of memory
# max_iters = 10000
# learning_rate = 6e-4  # Lower LR for larger models

# # CPU ONLY (slow but works everywhere)
# device = 'cpu'
# compile = False
# batch_size = 16
# dtype = 'float32'
# max_iters = 2000

# # HIGH QUALITY (long training for best results)
# max_iters = 20000
# lr_decay_iters = 20000
# eval_interval = 500
# n_layer = 8
# n_head = 8
# n_embd = 512

# ==============================================================================
# NOTES AND TIPS
# ==============================================================================

"""
TRAINING TIPS:

1. Model Size vs Quality vs Speed:
   - Smaller models (4 layers, 256 dim) train faster but lower quality
   - Larger models (8+ layers, 512+ dim) give better results but need more compute
   - The default 6 layer, 384 dim model is a good balance

2. Batch Size:
   - Larger batch sizes generally give more stable training
   - If you get out of memory errors, reduce batch_size
   - Use gradient_accumulation_steps to simulate larger batches

3. Learning Rate:
   - If loss doesn't decrease, try lower learning rate (1e-4)
   - If training is very slow, try higher learning rate (3e-3)
   - Use warmup for stability at the start

4. Context Length (block_size):
   - 256 is good for Shakespeare (captures sentence structure)
   - 512+ captures longer dependencies but needs more memory
   - Longer context = more memory usage (scales quadratically)

5. Training Duration:
   - Watch validation loss - stop when it stops improving
   - Shakespeare typically needs 3000-10000 iterations
   - More data needs more iterations

6. Hardware Optimization:
   - Use GPU if available (10-50x faster than CPU)
   - Enable mixed precision (dtype='float16') for speed
   - Use compile=True with PyTorch 2.0 for extra speed
   - Multiple GPUs can train larger models faster

7. Monitoring Training:
   - Watch for overfitting (train loss << val loss)
   - Good models have train loss ≈ val loss
   - Enable wandb_log=True for detailed tracking

8. Common Issues:
   - Loss = NaN: Lower learning rate or enable gradient clipping
   - Out of memory: Reduce batch_size or n_embd
   - Slow convergence: Increase learning rate or model size
   - Poor quality: More training iterations or larger model
"""
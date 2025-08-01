"""
Shakespeare Model Training Script
Trains a nanoGPT model on Shakespeare's complete works.
"""

import os
import sys
import time
import math
import pickle
import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_utils import GPT, GPTConfig

# Default configuration
config = {
    # I/O
    'out_dir': 'models/shakespeare_model',
    'eval_interval': 250,
    'log_interval': 10,
    'eval_iters': 200,
    'eval_only': False,  # if True, script exits right after the first eval
    'always_save_checkpoint': True,  # if True, always save a checkpoint after each eval
    'init_from': 'scratch',  # 'scratch' or 'resume' or 'gpt2*'
    
    # wandb logging
    'wandb_log': False,  # disabled by default
    'wandb_project': 'shakespeare-nanogpt',
    'wandb_run_name': 'gpt2',  # 'run' + str(time.time())
    
    # data
    'dataset': 'shakespeare',
    'gradient_accumulation_steps': 1,  # used to simulate larger batch sizes
    'batch_size': 64,  # if gradient_accumulation_steps > 1, this is the micro-batch size
    'block_size': 256,
    
    # model
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 384,
    'dropout': 0.2,  # for pretraining 0 is good, for finetuning try 0.1+
    'bias': False,  # do we use bias inside LayerNorm and Linear layers?
    
    # adamw optimizer
    'learning_rate': 1e-3,  # max learning rate
    'max_iters': 5000,  # total number of training iterations
    'weight_decay': 1e-1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,  # clip gradients at this value, or disable if == 0.0
    
    # learning rate decay settings
    'decay_lr': True,  # whether to decay the learning rate
    'warmup_iters': 100,  # how many steps to warm up for
    'lr_decay_iters': 5000,  # should be ~= max_iters per Chinchilla
    'min_lr': 1e-4,  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    
    # DDP settings
    'backend': 'nccl',  # 'nccl', 'gloo', etc.
    
    # system
    'device': 'auto',  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    'compile': True,  # use PyTorch 2.0 to compile the model to be faster
}

def get_batch(split, train_data, val_data, device, block_size, batch_size):
    """Generate a small batch of data of inputs x and targets y"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device != 'cpu':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, block_size, batch_size, eval_iters, ctx):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, device, block_size, batch_size)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, config):
    """Learning rate decay scheduler (cosine with warmup)"""
    # 1) linear warmup for warmup_iters steps
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config['lr_decay_iters']:
        return config['min_lr']
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

def train_model(config_overrides=None):
    """Main training function"""
    
    # Override config with any provided values
    if config_overrides:
        config.update(config_overrides)
    
    print("🎭 Shakespeare nanoGPT Training")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # Setup device
    if config['device'] == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config['device']
    
    print(f"🖥️  Using device: {device}")
    
    # Setup data type and context
    dtype = config['dtype']
    if dtype == 'float32':
        ptdtype = torch.float32
        ctx = nullcontext()
    elif dtype == 'bfloat16':
        ptdtype = torch.bfloat16
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
    elif dtype == 'float16':
        ptdtype = torch.float16
        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if device == 'cuda' else nullcontext()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    print(f"🔢 Using dtype: {dtype}")
    
    # Load data
    data_dir = Path('data/shakespeare')
    
    # Load metadata
    meta_path = data_dir / 'meta.pkl'
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}. Run 'make prepare-data' first.")
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    vocab_size = meta['vocab_size']
    print(f"📚 Vocabulary size: {vocab_size}")
    
    # Load training data
    train_data = np.fromfile(data_dir / 'train.bin', dtype=np.uint16)
    val_data = np.fromfile(data_dir / 'val.bin', dtype=np.uint16)
    
    print(f"🚂 Train data: {len(train_data):,} tokens")
    print(f"✅ Validation data: {len(val_data):,} tokens")
    
    # Initialize model
    print("\n🏗️  Initializing model...")
    
    model_args = dict(
        n_layer=config['n_layer'], 
        n_head=config['n_head'], 
        n_embd=config['n_embd'], 
        block_size=config['block_size'],
        bias=config['bias'], 
        vocab_size=vocab_size, 
        dropout=config['dropout']
    )
    
    if config['init_from'] == 'scratch':
        print("🆕 Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif config['init_from'] == 'resume':
        print(f"🔄 Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # Force these config attributes to be equal otherwise we can't even resume training
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        raise ValueError(f"Unknown init_from: {config['init_from']}")
    
    model.to(device)
    
    # Initialize variables for resuming
    if config['init_from'] == 'scratch':
        iter_num = 0
        best_val_loss = 1e9
    
    # Initialize a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Optimizer
    optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device)
    if config['init_from'] == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # free up memory
    
    # Compile the model
    if config['compile']:
        print("🔧 Compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0
    
    # Wrap model for multi-GPU training if available
    raw_model = model
    
    print(f"\n🎯 Training configuration:")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Block size: {config['block_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Max iterations: {config['max_iters']}")
    print(f"   Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"💡 Training will take approximately {config['max_iters'] * config['batch_size'] / 1000:.0f}k tokens")
    
    # Setup logging file
    log_file = os.path.join(config['out_dir'], 'training.log')
    with open(log_file, 'w') as f:
        f.write("iter,train_loss,val_loss,lr,time\n")
    
    X, Y = get_batch('train', train_data, val_data, device, config['block_size'], config['batch_size'])  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0
    
    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num, config) if config['decay_lr'] else config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % config['eval_interval'] == 0:
            losses = estimate_loss(model, train_data, val_data, device, config['block_size'], config['batch_size'], config['eval_iters'], ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log to file
            with open(log_file, 'a') as f:
                f.write(f"{iter_num},{losses['train']:.4f},{losses['val']:.4f},{lr:.6f},{time.time()-t0:.2f}\n")
            
            if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"💾 Saving checkpoint to {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
        
        if iter_num == 0 and config['eval_only']:
            break
        
        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(config['gradient_accumulation_steps']):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / config['gradient_accumulation_steps']  # scale the loss to account for gradient accumulation
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train', train_data, val_data, device, config['block_size'], config['batch_size'])
            # Backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        
        # Clip the gradient
        if config['grad_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        # Step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        
        # Flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % config['log_interval'] == 0:
            # Get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * config['gradient_accumulation_steps']
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        iter_num += 1
        local_iter_num += 1
        
        # Termination conditions
        if iter_num > config['max_iters']:
            break
    
    # Final checkpoint
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"💾 Saving final checkpoint to {config['out_dir']}")
    torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
    
    print(f"\n🎉 Training completed!")
    print(f"   Final validation loss: {best_val_loss:.4f}")
    print(f"   Total iterations: {iter_num}")
    print(f"   Model saved to: {config['out_dir']}")
    print(f"   Training log: {log_file}")

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Train Shakespeare nanoGPT model')
    
    # Model architecture
    parser.add_argument('--n_layer', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=256, help='Context length')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=5000, help='Maximum training iterations')
    parser.add_argument('--warmup_iters', type=int, default=100, help='Warmup iterations')
    parser.add_argument('--eval_interval', type=int, default=250, help='Evaluation interval')
    
    # System
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--compile', type=bool, default=True, help='Compile model with PyTorch 2.0')
    parser.add_argument('--out_dir', type=str, default='models/shakespeare_model', help='Output directory')
    
    # Resume training
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config file if provided
    config_overrides = {}
    if args.config:
        print(f"📄 Loading config from {args.config}")
        # Simple config loading (you can make this more sophisticated)
        exec(open(args.config).read(), {'__builtins__': {}}, config_overrides)
    
    # Override with command line arguments
    for arg in vars(args):
        if getattr(args, arg) is not None and arg != 'config':
            if arg == 'resume' and getattr(args, arg):
                config_overrides['init_from'] = 'resume'
            else:
                config_overrides[arg] = getattr(args, arg)
    
    # Start training
    try:
        train_model(config_overrides)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n💥 Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
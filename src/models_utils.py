"""
Shakespeare Model Utilities
Handles model loading, text generation, and inference utilities.
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse

class GPTConfig:
    """Configuration for the GPT model"""
    def __init__(self, **kwargs):
        self.block_size = 1024
        self.vocab_size = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.0
        self.bias = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        
        # Update with any provided kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """Multi-layer perceptron"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """GPT Language Model"""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for argument 'parameters'"
        # making the weight sharing explicit. not 100% sure this is the right way to do it, but it works.
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class ShakespeareModel:
    """High-level interface for the Shakespeare model"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.stoi = None
        self.itos = None
        self.config = {
            'temperature': 0.8,
            'top_k': 200,
            'max_length': 100
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            # Load metadata (character mappings)
            meta_path = os.path.join(os.path.dirname(self.model_path), '../data/shakespeare/meta.pkl')
            if not os.path.exists(meta_path):
                meta_path = 'data/shakespeare/meta.pkl'
                
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.stoi = meta['stoi']
            self.itos = meta['itos']
            
            # Load model checkpoint
            ckpt_path = os.path.join(self.model_path, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            # Create model with correct config
            model_config = GPTConfig(**checkpoint['model_args'])
            self.model = GPT(model_config)
            
            # Load state dict
            state_dict = checkpoint['model']
            # Handle potential key mismatches
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            
            print(f"Model loaded successfully! Running on {self.device}")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def encode(self, text):
        """Encode text to integers"""
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens):
        """Decode integers to text"""
        return ''.join([self.itos[i] for i in tokens])
    
    def generate_response(self, prompt, max_length=None, temperature=None):
        """Generate a response to a prompt"""
        if max_length is None:
            max_length = self.config['max_length']
        if temperature is None:
            temperature = self.config['temperature']
            
        # Encode the prompt
        encoded = self.encode(prompt)
        x = torch.tensor(encoded, dtype=torch.long, device=self.device)[None, ...]
        
        # Generate
        with torch.no_grad():
            y = self.model.generate(x, max_length, temperature=temperature, top_k=self.config['top_k'])
            generated = self.decode(y[0].tolist())
            
        # Return only the new part (after the prompt)
        return generated[len(prompt):]
    
    def generate_sample(self, max_length=200):
        """Generate a sample text starting with newline"""
        start = "\n"
        return self.generate_response(start, max_length)
    
    def get_config(self):
        """Get current generation configuration"""
        return self.config.copy()
    
    def update_config(self, new_config):
        """Update generation configuration"""
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value

def test_model():
    """Test model loading and generation"""
    try:
        model_path = 'models/shakespeare_model'
        shakespeare = ShakespeareModel(model_path)
        
        print("🧪 Testing model generation...")
        sample = shakespeare.generate_sample(max_length=100)
        print("Sample text:")
        print(sample)
        print("\n✅ Model test passed!")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Shakespeare Model Utilities')
    parser.add_argument('--mode', choices=['test', 'sample', 'chat'], default='sample',
                       help='Mode to run: test, sample, or chat')
    parser.add_argument('--model_path', default='models/shakespeare_model',
                       help='Path to model directory')
    parser.add_argument('--prompt', default='ROMEO:',
                       help='Prompt for generation')
    parser.add_argument('--max_length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Generation temperature')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_model()
    elif args.mode == 'sample':
        try:
            shakespeare = ShakespeareModel(args.model_path)
            sample = shakespeare.generate_sample(args.max_length)
            print("Generated sample:")
            print(sample)
        except Exception as e:
            print(f"Error: {e}")
    elif args.mode == 'chat':
        try:
            shakespeare = ShakespeareModel(args.model_path)
            print("🎭 Shakespeare Chatbot (type 'quit' to exit)")
            while True:
                prompt = input("\nYou: ")
                if prompt.lower() == 'quit':
                    break
                response = shakespeare.generate_response(prompt, args.max_length, args.temperature)
                print(f"Shakespeare: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    import math  # Add missing import
    main()
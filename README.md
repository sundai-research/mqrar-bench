# Standalone Models

Pure PyTorch implementations of linear attention and transformer models using FLA kernels.

## Quick Start

```python
from models import DeltaNet, Transformer

# Create a DeltaNet model
model = DeltaNet(
    vocab_size=32000,
    hidden_size=1024,
    num_layers=24,
    num_heads=4
)

# Or a Transformer
model = Transformer(
    vocab_size=32000,
    hidden_size=2048,
    num_layers=24,
    num_heads=32
)

# Standard PyTorch training
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for batch in dataloader:
    input_ids = batch['input_ids'].cuda()
    labels = batch['labels'].cuda()

    optimizer.zero_grad()
    logits, loss = model(input_ids, labels)
    loss.backward()
    optimizer.step()
```

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
standalone_models/
├── models/                  # Model implementations
│   ├── __init__.py         # Import all models here
│   ├── deltanet.py         # DeltaNet model
│   └── transformer.py      # Transformer with RoPE
├── examples/               # Training examples
│   ├── train_deltanet.py
│   └── train_transformer.py
├── requirements.txt        # All dependencies
└── README.md              # This file
```

## Available Models

### DeltaNet
Linear attention model with O(n) complexity instead of O(n²).

**Features:**
- Uses delta rule for efficient parallelization
- Memory efficient for long sequences
- Fast training with chunk mode
- Pure Triton kernels from FLA

**Paper:** [Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484)

**Usage:**
```python
from models import DeltaNet

model = DeltaNet(
    vocab_size=32000,
    hidden_size=1024,
    num_layers=24,
    num_heads=4,
    mode='chunk'  # or 'fused_recurrent' for inference
).cuda()
```

### Transformer
Standard decoder-only transformer with RoPE (Llama-style architecture).

**Features:**
- RoPE positional embeddings
- Grouped-Query Attention (GQA) support
- SwiGLU activations
- Flash Attention 2 support
- RMSNorm

**Usage:**
```python
from models import Transformer

model = Transformer(
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,  # For GQA (optional)
).cuda()
```

## Training Examples

### DeltaNet Training

```bash
python examples/train_deltanet.py \
    --hidden_size 512 \
    --num_layers 6 \
    --batch_size 8 \
    --epochs 10
```

### Transformer Training

```bash
python examples/train_transformer.py \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 4 \
    --compile  # Optional: use torch.compile
```

## Export to Your Project

This entire folder is self-contained and can be copied to any project:

```bash
# Copy to your project
cp -r standalone_models /path/to/your/project/

# Install dependencies
cd /path/to/your/project/standalone_models
pip install -r requirements.txt

# Use the models
from models import DeltaNet, Transformer
```

## Model Comparison

| Model | Attention | Complexity | Memory | Best For |
|-------|-----------|------------|--------|----------|
| **DeltaNet** | Linear | O(n) | Low | Long sequences (>8k tokens) |
| **Transformer** | Quadratic | O(n²) | High | Standard tasks (<8k tokens) |

## Citation

```bibtex
@software{yang2024fla,
  title  = {FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism},
  author = {Yang, Songlin and Zhang, Yu},
  url    = {https://github.com/fla-org/flash-linear-attention},
  year   = {2024}
}
```

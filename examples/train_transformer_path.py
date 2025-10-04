"""
Example training script for PaTH Attention Transformer

This demonstrates a minimal PyTorch training loop.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

from models import PaTHTransformer


class DummyDataset(Dataset):
    """Dummy dataset for demonstration"""
    def __init__(self, vocab_size=32000, seq_len=256, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random tokens
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {'input_ids': tokens, 'labels': tokens}


def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits, loss = model(input_ids, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train PaTH Attention Transformer')
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_kv_heads', type=int, default=None, help='For GQA')
    parser.add_argument('--use_forget_gate', action='store_true', help='Use forget gates')
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    args = parser.parse_args()

    print(f"Training PaTH Attention Transformer on {args.device}")
    print(f"Model config: {args.hidden_size}d x {args.num_layers}L x {args.num_heads}H")

    # Create model
    model = PaTHTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        use_forget_gate=args.use_forget_gate,
    ).to(args.device)

    # Optional: compile with PyTorch 2.0
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create dataset and dataloader
    dataset = DummyDataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_samples=1000
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, args.device)
        print(f"Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoint_path_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("\nTraining completed!")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    prompt = torch.randint(0, args.vocab_size, (1, 10)).to(args.device)
    generated = model.generate(prompt, max_length=50, temperature=0.8, top_k=40)
    print(f"Generated tokens: {generated[0].tolist()}")


if __name__ == '__main__':
    main()

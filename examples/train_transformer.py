"""
Example training script for Transformer with RoPE

This demonstrates a minimal PyTorch training loop.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

from models import Transformer


def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)

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
    parser = argparse.ArgumentParser(description='Train Transformer with RoPE')
    parser.add_argument('--vocab_size', type=int, default=65)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--intermediate_size', type=int, default=4*256)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_kv_heads', type=int, default=None, help='For GQA')
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    args = parser.parse_args()

    print(f"Training Transformer on {args.device}")
    print(f"Model config: {args.hidden_size}d x {args.num_layers}L x {args.num_heads}H")

    # Create model
    model = Transformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        max_position_embeddings=args.seq_len,
        dropout_value=0.1,
    ).to(args.device)

    # Optional: compile with PyTorch 2.0
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    from data_gen import SyntheticData
    from torch.utils.data import TensorDataset
    data = torch.load('data_baseline.pt', weights_only=False)

    dataloader = DataLoader(
        TensorDataset(data.train_inputs, data.train_labels),
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
    )
    test_dl = DataLoader(
        TensorDataset(data.test_inputs, data.test_labels),
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
    )

    # Optimizer (Llama-style)
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
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("\nTraining completed!")

    # Test generation
    print("\nEvaluating...")
    model.eval()
    generated_labels = []
    true_labels = []
    for batch_inputs, batch_labels in test_dl:
        input_ids = batch_inputs.to(args.device)
        labels = batch_labels.to(args.device)
        logits, loss = model(input_ids, labels)

        generated_labels.append(logits.argmax(dim=-1))
        true_labels.append(labels.to('cpu'))

    generated_labels = torch.cat(generated_labels, dim=0)
    true_labels = torch.cat(true_labels, dim=0)

    # Only calculate accuracy for non-ignored positions (-100)
    valid_mask = true_labels != -100
    correct = (generated_labels.to('cpu') == true_labels) & valid_mask
    accuracy = correct.sum().float() / valid_mask.sum().float()
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()

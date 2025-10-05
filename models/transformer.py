"""
Standalone Transformer Model with RoPE
Pure PyTorch implementation using Flash Attention and FLA's RoPE
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

# Import FLA's optimized RoPE and RMSNorm
from fla.modules import RMSNorm, RotaryEmbedding

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash-attn not installed. Falling back to PyTorch scaled_dot_product_attention")


class TransformerAttention(nn.Module):
    """Multi-head attention with RoPE"""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        layer_idx: int = 0,
        dropout_value: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.layer_idx = layer_idx

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # RoPE
        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            base=rope_theta
        )
        self.dropout_value = dropout_value if self.training else 0.0
        self.dropout = nn.Dropout(self.dropout_value)
    

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: Optional causal mask
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # Project to q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rotary(q, k)

        # Grouped-query attention (expand k, v if needed)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # Flash attention or PyTorch SDPA
        if HAS_FLASH_ATTN:
            # Flash Attention expects [batch, seq, heads, dim]
            attn_output = flash_attn_func(
                q, k, v,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim)
            )
        else:
            # PyTorch SDPA expects [batch, heads, seq, dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim)
            )
            attn_output = attn_output.transpose(1, 2)

        attn_output = self.dropout(attn_output)

        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        return output, None


class TransformerBlock(nn.Module):
    """Transformer block with RMSNorm and SwiGLU MLP"""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        intermediate_size: int = 8192,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        layer_idx: int = 0,
        dropout_value: float = 0.0,
    ):
        super().__init__()

        # Attention
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = TransformerAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            layer_idx=layer_idx,
            dropout_value=dropout_value,
        )

        # MLP
        self.mlp_norm = RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x)
        x = residual + x

        # MLP block (SwiGLU)
        residual = x
        x = self.mlp_norm(x)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = residual + x

        return x


class Transformer(nn.Module):
    """
    Standalone Transformer Language Model with RoPE

    This is a standard decoder-only transformer (similar to Llama/GPT)
    with RoPE positional embeddings and SwiGLU activations.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA, None for MHA)
        intermediate_size: MLP intermediate dimension
        max_position_embeddings: Maximum sequence length
        rope_theta: RoPE base frequency
        pad_token_id: Padding token ID
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        pad_token_id: int = 0,
        dropout_value: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        # Default intermediate size
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                layer_idx=i,
                dropout_value=dropout_value,
            )
            for i in range(num_layers)
        ])

        # Output
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights (similar to Llama)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len] - Input token IDs
            labels: [batch_size, seq_len] - Target token IDs (optional)

        Returns:
            logits: [batch_size, seq_len, vocab_size] - Output logits
            loss: Scalar loss (if labels provided)
        """
        # Embed inputs
        x = self.embeddings(input_ids)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100  # Use -100 to match data labels
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Simple generation with sampling

        Args:
            input_ids: [batch_size, seq_len] - Prompt tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (None for no filtering)
            top_p: Nucleus sampling (None for no filtering)

        Returns:
            generated: [batch_size, max_length] - Generated tokens
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                logits, _ = self.forward(input_ids)

                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k is not None:
                    topk_values, topk_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, topk_indices, topk_values)

                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

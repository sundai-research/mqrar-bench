"""
Standalone DeltaNet Model
Pure PyTorch implementation using FLA kernels
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Import only the kernel ops from FLA
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
from fla.modules import RMSNorm, ShortConvolution


class DeltaNetAttention(nn.Module):
    """DeltaNet attention layer using FLA kernels"""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_beta: bool = True,
        use_short_conv: bool = False,
        conv_size: int = 4,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.mode = mode
        self.use_gate = use_gate
        self.use_beta = use_beta
        self.use_short_conv = use_short_conv
        self.layer_idx = layer_idx

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_beta:
            self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        if use_short_conv:
            self.conv = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # Project to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            k = self.conv(k)

        # Apply activations (ELU+1 for keys, queries)
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1

        # Normalize
        k = k / k.sum(-1, keepdim=True)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_qk_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_qk_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_v_dim)

        # Beta (forgetting factor)
        if self.use_beta:
            beta = self.b_proj(x).view(batch_size, seq_len, self.num_heads)
            beta = torch.sigmoid(beta)
            # Reshape to (batch_size, num_heads, seq_len) as expected by FLA
            beta = beta.transpose(1, 2)
        else:
            beta = None

        # Ensure all tensors have the same dtype
        target_dtype = q.dtype
        k = k.to(target_dtype)
        v = v.to(target_dtype)
        if beta is not None:
            beta = beta.to(target_dtype)

        # Apply DeltaNet kernel
        if self.mode == 'chunk':
            result = chunk_delta_rule(q, k, v, beta)
            # Handle tuple return (o, final_state)
            if isinstance(result, tuple):
                o, _ = result
            else:
                o = result
        elif self.mode == 'fused_recurrent':
            result = fused_recurrent_delta_rule(q, k, v, beta)
            # Handle tuple return (o, final_state)
            if isinstance(result, tuple):
                o, _ = result
            else:
                o = result
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Reshape back
        o = o.reshape(batch_size, seq_len, self.value_dim)

        # Apply gate if used
        if self.use_gate:
            g = torch.sigmoid(self.g_proj(x))
            o = o * g

        # Output projection
        o = self.o_proj(o)

        return o, None


class DeltaNetBlock(nn.Module):
    """DeltaNet transformer block"""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        intermediate_size: int = 4096,
        mode: str = 'chunk',
        layer_idx: int = 0,
    ):
        super().__init__()

        # Attention
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = DeltaNetAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            expand_k=expand_k,
            expand_v=expand_v,
            mode=mode,
            layer_idx=layer_idx,
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


class DeltaNet(nn.Module):
    """
    Standalone DeltaNet Language Model

    Paper: Parallelizing Linear Transformers with the Delta Rule over Sequence Length
    https://arxiv.org/abs/2406.06484

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        expand_k: Key dimension expansion factor
        expand_v: Value dimension expansion factor
        intermediate_size: MLP intermediate dimension
        mode: Computation mode ('chunk' or 'fused_recurrent')
        pad_token_id: Padding token ID
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 4,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        intermediate_size: int = 4096,
        mode: str = 'chunk',
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        # Transformer blocks
        self.layers = nn.ModuleList([
            DeltaNetBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expand_k=expand_k,
                expand_v=expand_v,
                intermediate_size=intermediate_size,
                mode=mode,
                layer_idx=i,
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
        """Initialize weights"""
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
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Simple greedy/sampling generation

        Args:
            input_ids: [batch_size, seq_len] - Prompt tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)

        Returns:
            generated: [batch_size, max_length] - Generated tokens
        """
        self.eval()
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                logits, _ = self.forward(input_ids)

                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature

                # Sample or greedy
                if top_k is not None:
                    # Top-k sampling
                    topk_values, topk_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, topk_indices, topk_values)
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

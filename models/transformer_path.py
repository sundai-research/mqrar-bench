"""
Standalone PaTH Attention Model
Pure PyTorch implementation using FLA kernels

Paper: PaTH Attention: Position Encoding via Accumulating Householder Transformations
https://arxiv.org/abs/2505.16381
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# Import FLA kernels and modules
from fla.ops.path_attn.parallel import parallel_path_attn
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm


class PaTHAttention(nn.Module):
    """
    PaTH Attention: Position Encoding via Accumulating Householder Transformations

    This uses Householder transformations for position encoding instead of RoPE,
    which provides better interpolation and extrapolation properties.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        use_forget_gate: bool = False,
        use_qk_norm: bool = False,
        use_low_rank_w: bool = True,
        use_w_shortconv: bool = True,
        conv_size: int = 3,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.layer_idx = layer_idx

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)

        # W projection for Householder reflections
        if use_low_rank_w:
            self.w_proj = nn.Sequential(
                nn.Linear(hidden_size, 32, bias=False),
                nn.Linear(32, self.kv_dim, bias=False)
            )
        else:
            self.w_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)

        # Optional: QK normalization
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Optional: Short convolution on W
        self.use_w_shortconv = use_w_shortconv
        if use_w_shortconv:
            self.w_conv1d = ShortConvolution(
                hidden_size=self.kv_dim,
                kernel_size=conv_size,
                bias=False,
                activation='silu'
            )

        # Beta projection (for controlling transformation strength)
        self.bt_proj = nn.Linear(hidden_size, self.num_kv_heads, bias=True)

        # Optional: Forget gate
        self.use_forget_gate = use_forget_gate
        if use_forget_gate:
            self.g_proj = nn.Linear(hidden_size, self.num_heads, bias=True)

        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: Not used in training (for compatibility)
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # Project to q, k, v, w
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        w = self.w_proj(x)

        # Beta (transformation strength)
        beta = self.bt_proj(x).float().sigmoid() * 2  # Range [0, 2]

        # Optional forget gate
        g = F.logsigmoid(self.g_proj(x).float()) if self.use_forget_gate else None

        # Apply short convolution to w if enabled
        if self.use_w_shortconv:
            w, _ = self.w_conv1d(w, cache=None, output_final_state=False)

        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        w = w.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Optional QK normalization
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Normalize w (Householder reflections require unit vectors)
        w = l2_norm(w, output_dtype=torch.float32)

        # Apply PaTH attention kernel
        o, _ = parallel_path_attn(q=q, k=k, v=v, w=w, beta=beta, g=g)

        # Reshape and project output
        o = o.reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(o)

        return output, None


class PaTHBlock(nn.Module):
    """Transformer block with PaTH Attention"""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        use_forget_gate: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()

        # PaTH Attention
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = PaTHAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_forget_gate=use_forget_gate,
            layer_idx=layer_idx,
        )

        # MLP
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)

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


class PaTHTransformer(nn.Module):
    """
    Standalone PaTH Attention Transformer

    Paper: PaTH Attention: Position Encoding via Accumulating Householder Transformations
    https://arxiv.org/abs/2505.16381

    Unlike RoPE which rotates query/key vectors, PaTH uses Householder transformations
    which provides better interpolation/extrapolation to longer contexts.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA, None for MHA)
        intermediate_size: MLP intermediate dimension
        use_forget_gate: Whether to use forget gates
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
        use_forget_gate: bool = False,
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
            PaTHBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                use_forget_gate=use_forget_gate,
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

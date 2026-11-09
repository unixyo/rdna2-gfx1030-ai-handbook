"""Small, explicit RDNA2 inference operators; PyTorch handles other cases."""
from .normalization import GroupNormSiLU, group_norm_silu
from .attention import scaled_dot_product_attention

__all__ = ["group_norm_silu", "GroupNormSiLU", "scaled_dot_product_attention"]
__version__ = "0.7.0"

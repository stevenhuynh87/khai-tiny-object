"""Pure-Python fallback for the missing CUDA MultiScaleDeformableAttention extension."""

import sys
import types
from typing import Any

import torch
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: Any,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Run multi-scale deformable sampling with torch.grid_sample."""
    batch_size, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    if hasattr(value_spatial_shapes, "detach"):
        spatial_shapes = [
            (int(height), int(width))
            for height, width in value_spatial_shapes.detach().cpu().tolist()
        ]
    else:
        spatial_shapes = [(int(height), int(width)) for height, width in value_spatial_shapes]

    value_list = value.split(
        [height * width for height, width in spatial_shapes],
        dim=1,
    )
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampling_values: list[torch.Tensor] = []

    for level, (height, width) in enumerate(spatial_shapes):
        level_value = value_list[level].flatten(2).transpose(1, 2)
        level_value = level_value.reshape(
            batch_size * num_heads,
            head_dim,
            height,
            width,
        )
        level_grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        level_sample = F.grid_sample(
            level_value,
            level_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_values.append(level_sample)

    attention = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads,
        1,
        num_queries,
        num_levels * num_points,
    )
    output = (
        torch.stack(sampling_values, dim=-2).flatten(-2) * attention
    ).sum(-1)
    output = output.view(batch_size, num_heads * head_dim, num_queries)
    return output.transpose(1, 2).contiguous()


def ms_deform_attn_forward(
    value: torch.Tensor,
    value_spatial_shapes: Any,
    value_level_start_index: Any,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    """Fallback forward operator."""
    del value_level_start_index, im2col_step
    return ms_deform_attn_core_pytorch(
        value,
        value_spatial_shapes,
        sampling_locations,
        attention_weights,
    )


def ms_deform_attn_backward(
    value: torch.Tensor,
    value_spatial_shapes: Any,
    value_level_start_index: Any,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    grad_output: torch.Tensor,
    im2col_step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gradients for the pure-PyTorch fallback via autograd."""
    del value_level_start_index, im2col_step

    with torch.enable_grad():
        value_leaf = value.detach().requires_grad_(True)
        sampling_leaf = sampling_locations.detach().requires_grad_(True)
        attention_leaf = attention_weights.detach().requires_grad_(True)
        output = ms_deform_attn_core_pytorch(
            value_leaf,
            value_spatial_shapes,
            sampling_leaf,
            attention_leaf,
        )
        gradients = torch.autograd.grad(
            output,
            (value_leaf, sampling_leaf, attention_leaf),
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
    grad_value = gradients[0] if gradients[0] is not None else torch.zeros_like(value)
    grad_sampling = (
        gradients[1]
        if gradients[1] is not None
        else torch.zeros_like(sampling_locations)
    )
    grad_attention = (
        gradients[2]
        if gradients[2] is not None
        else torch.zeros_like(attention_weights)
    )
    return grad_value, grad_sampling, grad_attention


def install_fallback() -> None:
    """Register a pure-PyTorch module under the CUDA extension module name."""
    fallback = types.ModuleType("MultiScaleDeformableAttention")
    fallback.ms_deform_attn_forward = ms_deform_attn_forward
    fallback.ms_deform_attn_backward = ms_deform_attn_backward
    sys.modules["MultiScaleDeformableAttention"] = fallback

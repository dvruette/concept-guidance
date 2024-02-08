import functools
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import safetensors


def load_weights(path: str | Path, device="cpu"):
    with safetensors.safe_open(path, framework="pt") as f:
        try:
            return f.get_tensor("W").to(device)
        except KeyError as e:
            raise ValueError(f"Incompatible weights file: {path}. Make sure the file has a `W` key.") from e
    

def apply_linear_guidance(x: torch.Tensor, w: torch.Tensor, guidance_scale=1.0, only_last=False, normalize=True):
    input_dtype = x.dtype
    w = w.to(x.device, non_blocking=True)
    x = x.to(torch.float32)
    w = w / w.norm(dim=-1, keepdim=True)

    x_norm = x.norm(dim=-1, keepdim=True)
    if only_last:
        x[:, -1:] += guidance_scale * w
    else:
        x += guidance_scale * w

    if normalize:
        x = x * (x_norm / x.norm(dim=-1, keepdim=True))
    return x.to(input_dtype)


def guidance_patch(
    self,
    vectors: torch.Tensor,
    layer_idx: int,
    ###
    x: torch.Tensor,
    guidance_scale=1.0,
    normalize=True,
    only_last=False
):
    x = self.old_forward(x)
    if guidance_scale == 0.0:
        return x

    w = vectors[layer_idx].unsqueeze(0).unsqueeze(0)
    return apply_linear_guidance(x, w, guidance_scale=guidance_scale, only_last=only_last, normalize=normalize)


def attn_guidance_patch(
    self,
    vectors: torch.Tensor,
    layer_idx: int,
    ###
    *args,
    guidance_scale=1.0,
    only_last=False,
    normalize=True,
    **kwargs
):
    outputs = self.old_forward(*args, **kwargs)
    if guidance_scale == 0.0:
        return outputs
    
    x = outputs[0]
    w = vectors[layer_idx].unsqueeze(0).unsqueeze(0)

    x = apply_linear_guidance(x, w, guidance_scale=guidance_scale, only_last=only_last, normalize=normalize)

    return (x,) + outputs[1:]


def patch_model(
    model: nn.Module,
    vectors: torch.Tensor,
    representation: Literal["pre-attn", "post-attn"] = "pre-attn",
    guidance_scale: float | list[float] = 1.0,
    guidance_layers: list[int] | list[tuple] | None = None,
    only_last: bool = False,
):
    num_layers = len(model.model.layers)
    if not num_layers == vectors.shape[0]:
        raise ValueError(f"Number of layers in model ({num_layers}) does not match number of concept vectors ({vectors.shape[0]})")
    
    if isinstance(guidance_scale, list):
        if len(guidance_scale) != vectors.shape[0]:
            raise ValueError(f"Number of guidance scales ({len(guidance_scale)}) does not match number of concept vectors ({vectors.shape[0]})")
    else:
        guidance_scale = [guidance_scale] * num_layers

    guidance_layers = [(idx,) if isinstance(idx, int) else idx for idx in guidance_layers]

    device = next(model.parameters()).device
    vectors = vectors.to(device, non_blocking=True)

    if representation in ("pre-attn",):
        for idx, layer in enumerate(model.model.layers):
            if guidance_layers is not None and (idx,) not in guidance_layers:
                continue
            ln = layer.input_layernorm
            ln.old_forward = ln.forward
            ln.forward = functools.partial(
                guidance_patch, ln, vectors, idx,
                guidance_scale=guidance_scale[idx],
                normalize=True,
                only_last=only_last,
            )
    elif representation in ("post-attn",):
        for idx, layer in enumerate(model.model.layers):
            if guidance_layers is not None and (idx,) not in guidance_layers:
                continue
            attn = layer.self_attn
            attn.old_forward = attn.forward
            attn.forward = functools.partial(
                attn_guidance_patch, attn, vectors, idx,
                guidance_scale=guidance_scale[idx],
                only_last=only_last,
            )
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    return model


def unpatch_model(model):
    for layer in model.model.layers:
        ln = layer.input_layernorm
        if hasattr(ln, "old_forward"):
            ln.forward = ln.old_forward
            del ln.old_forward

        attn = layer.self_attn
        if hasattr(attn, "old_forward"):
            attn.forward = attn.old_forward
            del attn.old_forward
    return model

"""
LoRA Utilities for I2V Fine-tuning

Provides utilities for applying LoRA to the I2V LatentVisualDiffusion model.
Adapted from t2v/lora_training/lora_utils.py with I2V-specific additions.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_target_modules(model: nn.Module, target_names: List[str] = None) -> List[str]:
    """
    Find all module names matching target patterns.

    Args:
        model: PyTorch model
        target_names: Patterns to match (e.g. ["to_q", "to_k"])

    Returns:
        List of fully qualified module names
    """
    if target_names is None:
        target_names = ["to_q", "to_k", "to_v", "to_out"]

    matched = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_names:
                if target in name:
                    matched.append(name)
                    break

    return matched


def get_lora_config(
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
    bias: str = "none",
):
    """Create a PEFT LoRA configuration."""
    try:
        from peft import LoraConfig
    except ImportError:
        raise ImportError("PEFT library not found. Install with: pip install peft")

    if target_modules is None:
        target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )

    return config


def apply_lora_to_model(model: nn.Module, lora_config) -> nn.Module:
    """Apply LoRA adapters to a model using PEFT."""
    try:
        from peft import get_peft_model
    except ImportError:
        raise ImportError("PEFT library not found. Install with: pip install peft")

    peft_model = get_peft_model(model, lora_config)
    return peft_model


def prepare_i2v_model_for_lora(
    model: nn.Module,
    freeze_vae: bool = True,
    freeze_text_encoder: bool = True,
    freeze_image_embedder: bool = True,
    freeze_image_proj: bool = True,
) -> nn.Module:
    """
    Prepare an I2V LatentVisualDiffusion model for LoRA training.

    Freezes all components except the UNet backbone:
      - VAE (first_stage_model)
      - Text encoder (cond_stage_model / FrozenOpenCLIPEmbedder)
      - Image embedder (embedder / FrozenOpenCLIPImageEmbedderV2)
      - Image projection model (image_proj_model / Resampler)
    """
    # Freeze VAE
    if freeze_vae and hasattr(model, 'first_stage_model'):
        for param in model.first_stage_model.parameters():
            param.requires_grad = False
        model.first_stage_model.eval()
        print("Froze VAE (first_stage_model)")

    # Freeze text encoder
    if freeze_text_encoder and hasattr(model, 'cond_stage_model'):
        for param in model.cond_stage_model.parameters():
            param.requires_grad = False
        model.cond_stage_model.eval()
        print("Froze text encoder (cond_stage_model)")

    # Freeze image embedder (CLIP vision)
    if freeze_image_embedder and hasattr(model, 'embedder'):
        for param in model.embedder.parameters():
            param.requires_grad = False
        model.embedder.eval()
        print("Froze image embedder (embedder)")

    # Freeze image projection model (Resampler)
    if freeze_image_proj and hasattr(model, 'image_proj_model'):
        for param in model.image_proj_model.parameters():
            param.requires_grad = False
        model.image_proj_model.eval()
        print("Froze image projection model (image_proj_model)")

    return model


def get_trainable_parameters(model: nn.Module) -> Dict[str, Any]:
    """Get count of trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total if total > 0 else 0

    return {
        'trainable': trainable,
        'total': total,
        'percentage': percentage,
        'trainable_mb': trainable * 4 / (1024 * 1024),
    }


def save_lora_weights(model: nn.Module, save_path: str):
    """Save LoRA weights."""
    try:
        from peft import PeftModel
    except ImportError:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "model.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved full model to {save_path}")
        return

    # Check if model.model is wrapped with PEFT
    if hasattr(model, 'model') and isinstance(model.model, PeftModel):
        model.model.save_pretrained(save_path)
        print(f"Saved LoRA weights to {save_path}")
    elif isinstance(model, PeftModel):
        model.save_pretrained(save_path)
        print(f"Saved LoRA weights to {save_path}")
    else:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "model.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Saved full model to {save_path}")


def load_lora_weights(model: nn.Module, lora_path: str) -> nn.Module:
    """Load LoRA weights into a model."""
    try:
        from peft import PeftModel
    except ImportError:
        model.load_state_dict(torch.load(lora_path), strict=False)
        return model

    model = PeftModel.from_pretrained(model, lora_path)
    print(f"Loaded LoRA weights from {lora_path}")
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into the base model (removes adapters)."""
    try:
        from peft import PeftModel
    except ImportError:
        return model

    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
        print("Merged LoRA weights into base model")
    return model

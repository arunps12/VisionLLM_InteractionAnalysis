# =========================================================
# UTILS: ML UTILS
# File: visionllm_interaction/utils/ml_utils.py
# =========================================================

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from visionllm_interaction.logger.logger import get_logger
from visionllm_interaction.exception.custom_exception import CustomException

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # deterministic options (can reduce speed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(f"Seed set to: {seed}")
    except Exception as e:
        raise CustomException("Failed to set random seed", e)


def get_device(device_cfg: str = "auto") -> torch.device:
    """
    Returns torch.device based on config.
    device_cfg:
      - "auto" -> cuda if available else cpu
      - "cuda" -> cuda if available else cpu (warn)
      - "cpu"  -> cpu
    """
    try:
        if device_cfg == "cpu":
            return torch.device("cpu")

        if device_cfg in ("auto", "cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA not available. Falling back to CPU.")
            return torch.device("cpu")

        # explicit device string
        return torch.device(device_cfg)

    except Exception as e:
        raise CustomException(f"Failed to resolve device from config: {device_cfg}", e)


def collate_fn(batch):
    """
    Detection models in torchvision expect:
      images: List[Tensor]
      targets: List[Dict]
    """
    return tuple(zip(*batch))


def ensure_dir(dir_path: str) -> None:
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        raise CustomException(f"Failed to create directory: {dir_path}", e)


def save_checkpoint(model: torch.nn.Module, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model checkpoint.
    """
    try:
        payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
        if extra:
            payload.update(extra)

        ensure_dir(os.path.dirname(path))
        torch.save(payload, path)
        logger.info(f"Saved checkpoint: {path}")

    except Exception as e:
        raise CustomException(f"Failed to save checkpoint: {path}", e)


def load_checkpoint(model: torch.nn.Module, path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load model checkpoint and restore model state dict.
    Returns checkpoint payload.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=map_location)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {path}")
        return ckpt

    except Exception as e:
        raise CustomException(f"Failed to load checkpoint: {path}", e)

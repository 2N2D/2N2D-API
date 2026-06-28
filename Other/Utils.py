import torch
import logging
import numpy as np

def get_device():
    """Detect and return the best available device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  
        logging.info(f"✓ GPU detected: {gpu_name} with {gpu_memory:.1f}GB memory")
        logging.info(f"Using GPU acceleration for optimization")
        return device
    else:
        logging.info("No GPU detected, using CPU for optimization")
        return torch.device("cpu")

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return list(make_json_serializable(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):
        return obj
    elif hasattr(obj, 'item'): 
        return obj.item()
    else:
        return obj

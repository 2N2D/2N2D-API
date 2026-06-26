import torch
import torch.nn as nn
from typing import Dict, Any, Type

class NodeRegistry:
    """Registry that maps textual node types to PyTorch operations and semantics."""
    
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, node_type: str, module_cls: Type[nn.Module], param_mapping: Dict[str, str], execution_op: str = "standard_forward"):
        cls._registry[node_type] = {
            "module_cls": module_cls,
            "param_mapping": param_mapping,
            "execution_op": execution_op
        }

    @classmethod
    def get_module(cls, node_type: str, params: Dict[str, Any]) -> nn.Module:
        if node_type not in cls._registry:
            # Fallback mappings for standard types if not explicitly registered
            cls._register_defaults()
            
        if node_type not in cls._registry:
            raise ValueError(f"Unknown node type: {node_type}")

        registry_info = cls._registry[node_type]
        module_cls = registry_info["module_cls"]
        param_mapping = registry_info["param_mapping"]

        kwargs = {}
        for param_key, param_value in params.items():
            if param_key in param_mapping:
                kwargs[param_mapping[param_key]] = param_value
            else:
                kwargs[param_key] = param_value

        return module_cls(**kwargs)

    @classmethod
    def get_execution_semantic(cls, node_type: str) -> str:
        if node_type not in cls._registry:
             cls._register_defaults()
        if node_type not in cls._registry:
            return "standard_forward"
        return cls._registry[node_type].get("execution_op", "standard_forward")

    @classmethod
    def _register_defaults(cls):
        # Prevent duplicate registrations
        if "Linear" in cls._registry:
            return
            
        cls.register("Linear", nn.Linear, {"in_features": "in_features", "out_features": "out_features", "bias": "bias"})
        cls.register("Conv1d", nn.Conv1d, {"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size", "stride": "stride"})
        cls.register("Conv2d", nn.Conv2d, {"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size", "stride": "stride"})
        cls.register("LSTM", nn.LSTM, {"input_size": "input_size", "hidden_size": "hidden_size", "num_layers": "num_layers", "batch_first": "batch_first"}, execution_op="recurrent_lstm")
        cls.register("GRU", nn.GRU, {"input_size": "input_size", "hidden_size": "hidden_size", "num_layers": "num_layers", "batch_first": "batch_first"}, execution_op="recurrent_gru")
        cls.register("ReLU", nn.ReLU, {})
        cls.register("Tanh", nn.Tanh, {})
        cls.register("Sigmoid", nn.Sigmoid, {})
        cls.register("Softmax", nn.Softmax, {})
        cls.register("Dropout", nn.Dropout, {"p": "p"})
        cls.register("Flatten", nn.Flatten, {"start_dim": "start_dim", "end_dim": "end_dim"})
        cls.register("BatchNorm1d", nn.BatchNorm1d, {"num_features": "num_features"})
        cls.register("BatchNorm2d", nn.BatchNorm2d, {"num_features": "num_features"})

# Initialize defaults
NodeRegistry._register_defaults()

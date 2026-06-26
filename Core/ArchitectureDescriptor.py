from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class Edge:
    source: str
    target: str
    source_port: str = "output"
    target_port: str = "input"

@dataclass
class Node:
    id: str
    type: str # Linear, Conv1D, Conv2D, LSTM, GRU, ReLU, etc.
    params: Dict[str, Any] = field(default_factory=dict)
    io: Dict[str, List[str]] = field(default_factory=lambda: {"inputs": ["input"], "outputs": ["output"]})
    execution: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TensorContract:
    shape: List[int]
    dtype: str = "float32"

@dataclass
class ArchitectureDescriptor:
    model_name: str
    input_shape: List[int]
    output_shape: List[int]
    nodes: List[Node]
    edges: List[Edge]
    tensor_contracts: Dict[str, TensorContract] = field(default_factory=dict)
    merge_rules: Dict[str, Any] = field(default_factory=dict)
    propagation_rules: str = "deterministic_forward"

    def validate(self):
        """Perform static validation of the descriptor."""
        node_ids = {node.id for node in self.nodes}
        
        # Check edges reference valid nodes
        for edge in self.edges:
            if edge.source not in node_ids and edge.source != "input":
                raise ValueError(f"Edge source not found: {edge.source}")
            if edge.target not in node_ids and edge.target != "output":
                raise ValueError(f"Edge target not found: {edge.target}")

        return True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchitectureDescriptor":
        nodes = [Node(**n) for n in data.get("nodes", [])]
        edges = [Edge(**e) for e in data.get("edges", [])]
        contracts = {
            k: TensorContract(**v) 
            for k, v in data.get("tensor_contracts", {}).items()
        }
        return cls(
            model_name=data["model_name"],
            input_shape=data["input_shape"],
            output_shape=data["output_shape"],
            nodes=nodes,
            edges=edges,
            tensor_contracts=contracts,
            merge_rules=data.get("merge_rules", {}),
            propagation_rules=data.get("propagation_rules", "deterministic_forward")
        )

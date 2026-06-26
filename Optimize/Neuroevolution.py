import copy
import random
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from Core.ArchitectureDescriptor import ArchitectureDescriptor, Node, Edge
from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.NodeRegistry import NodeRegistry

class MutationGrammar:
    @staticmethod
    def scale_width(descriptor: ArchitectureDescriptor, target_node_id: str, factor: float) -> bool:
        """Scales the width (hidden size, out_features, channels) of a node and propagates to adjacent."""
        node = next((n for n in descriptor.nodes if n.id == target_node_id), None)
        if not node: return False
        
        # Valid nodes for width scaling
        if node.type not in ["Linear", "Conv1d", "Conv2d", "LSTM", "GRU"]:
            return False

        old_out = 0
        param_to_scale = ""
        
        if node.type == "Linear":
            old_out = node.params.get("out_features", 0)
            param_to_scale = "out_features"
        elif "Conv" in node.type:
            old_out = node.params.get("out_channels", 0)
            param_to_scale = "out_channels"
        elif node.type in ["LSTM", "GRU"]:
            old_out = node.params.get("hidden_size", 0)
            param_to_scale = "hidden_size"
            
        if not old_out: return False
        
        new_out = max(1, int(old_out * factor))
        node.params[param_to_scale] = new_out

        # Downstream propagation
        downstream_targets = [e.target for e in descriptor.edges if e.source == target_node_id]
        
        for tgt_id in downstream_targets:
            tgt_node = next((n for n in descriptor.nodes if n.id == tgt_id), None)
            if not tgt_node: continue
            
            # Simple direct connection handling
            if tgt_node.type == "Linear":
                tgt_node.params["in_features"] = new_out
            elif "Conv" in tgt_node.type:
                tgt_node.params["in_channels"] = new_out
            elif tgt_node.type in ["LSTM", "GRU"]:
                tgt_node.params["input_size"] = new_out

        return True

    @staticmethod
    def add_layer(descriptor: ArchitectureDescriptor, after_node_id: str, new_node_type: str) -> bool:
        """Adds a layer after the specified node, maintaining contracts."""
        after_node = next((n for n in descriptor.nodes if n.id == after_node_id), None)
        if not after_node: return False

        # Find downstream edge
        target_edge = next((e for e in descriptor.edges if e.source == after_node_id), None)
        if not target_edge: return False

        new_node_id = f"{new_node_type.lower()}_add_{random.randint(1000, 9999)}"
        new_params = {}

        # Determine shapes based on after_node output
        out_dim = 64 # Default fallback
        if after_node.type == "Linear":
            out_dim = after_node.params.get("out_features", 64)
            
        if new_node_type == "Linear":
            new_params = {"in_features": out_dim, "out_features": out_dim}
        elif new_node_type in ["ReLU", "Dropout"]:
            new_params = {"p": 0.1} if new_node_type == "Dropout" else {}
        else:
            return False # Unsupported auto-add type for now

        new_node = Node(id=new_node_id, type=new_node_type, params=new_params)
        descriptor.nodes.append(new_node)

        # Reroute edges
        original_target = target_edge.target
        target_edge.target = new_node_id
        
        descriptor.edges.append(Edge(source=new_node_id, target=original_target))
        return True

class TieredEvaluator:
    @staticmethod
    def evaluate(model: nn.Module, data_loader, criterion, optimizer, num_epochs: int, device="cpu"):
        model.to(device)
        model.train()
        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                if outputs.dim() > batch_y.dim() and batch_y.dim() == 1:
                    outputs = outputs.squeeze(-1)
                elif outputs.dim() == 1 and batch_y.dim() == 2:
                    outputs = outputs.unsqueeze(-1)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(data_loader))
        return losses[-1] if losses else float('inf')

class NeuroevolutionEngine:
    def __init__(self, initial_descriptor: ArchitectureDescriptor, population_size: int = 20):
        self.initial_descriptor = initial_descriptor
        self.population_size = population_size
        self.population: filter[ArchitectureDescriptor] = []

    def initialize_population(self):
        self.population = [copy.deepcopy(self.initial_descriptor) for _ in range(self.population_size)]
        # Mutate to create diversity
        for desc in self.population[1:]:
            self._mutate_descriptor(desc)

    def _mutate_descriptor(self, desc: ArchitectureDescriptor):
        mut_type = random.choice(["scale_width", "add_layer"])
        nodes_to_mutate = [n.id for n in desc.nodes if n.type in ["Linear", "LSTM", "Conv1d"]]
        if not nodes_to_mutate: return
        
        target = random.choice(nodes_to_mutate)
        
        if mut_type == "scale_width":
            factor = random.choice([0.5, 0.8, 1.2, 1.5, 2.0])
            MutationGrammar.scale_width(desc, target, factor)
        elif mut_type == "add_layer":
            MutationGrammar.add_layer(desc, target, "ReLU" if random.random() > 0.5 else "Linear")

    def evolve(self, data_loader, generations: int = 5):
        self.initialize_population()
        
        for gen in range(generations):
            evaluated = []
            
            # Tier 1 Evaluation: 1 epoch for filtering
            for desc in self.population:
                try:
                    desc.validate()
                    model = DescriptorModelBuilder.build(desc)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    criterion = nn.MSELoss()
                    loss = TieredEvaluator.evaluate(model, data_loader, criterion, optimizer, num_epochs=1)
                    evaluated.append((desc, loss, model))
                except Exception as e:
                    # Invalid architecture or execution failed
                    pass
            
            # Sort by loss
            evaluated.sort(key=lambda x: x[1])
            
            # Selection
            top_k = evaluated[:max(2, self.population_size // 4)]
            new_population = [x[0] for x in top_k]
            
            # Crossover / Mutation for next gen
            while len(new_population) < self.population_size:
                parent = random.choice(top_k)[0]
                child = copy.deepcopy(parent)
                self._mutate_descriptor(child)
                new_population.append(child)
                
            self.population = new_population
            print(f"Generation {gen+1} complete. Best Loss: {top_k[0][1] if top_k else 'N/A'}")
            
        return self.population[0]

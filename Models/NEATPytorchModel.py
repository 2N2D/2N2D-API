import logging
import UniversalArchitectureModel
import torch.nn as nn
import neat

class NEATPytorchModel(nn.Module):
    def __init__(self, genome, config, input_size, output_size, base_model_info):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.base_model_info = base_model_info
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)
        self._build_enhanced_universal_architecture(genome)
        
    def _build_enhanced_universal_architecture(self, genome):
        """Build enhanced PyTorch architecture using universal approach"""
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        total_nodes = len(genome.nodes)
        baseline_connections = self.input_size + self.output_size  
        actual_connections = len(enabled_connections)
        connection_ratio = actual_connections / max(baseline_connections, 1)
        node_ratio = total_nodes / max(self.input_size + self.output_size + 2, 1)
        complexity_score = (connection_ratio * 2.0) + (node_ratio * 1.5)
        min_enhancement = 1.25
        max_enhancement = min(3.0, 1.0 + complexity_score)
        enhancement_factor = max(min_enhancement, max_enhancement)
        
        
        self._enhancement_factor = enhancement_factor
        
        logging.info(f"NEAT Enhancement Factor: {enhancement_factor:.2f} | "
                    f"Connections: {actual_connections} | Nodes: {total_nodes}")
        self.enhanced_model = UniversalArchitectureModel(
            architecture_info=self.base_model_info,
            enhancement_factor=enhancement_factor,
            actual_input_size=self.input_size,
            actual_output_size=self.output_size
        )
        enhanced_params = sum(p.numel() for p in self.enhanced_model.parameters())
        baseline_model = UniversalArchitectureModel(
            architecture_info=self.base_model_info,
            enhancement_factor=1.0,
            actual_input_size=self.input_size,
            actual_output_size=self.output_size
        )
        original_params = sum(p.numel() for p in baseline_model.parameters())
        
        if enhanced_params <= original_params:
            logging.warning(f"NEAT: Parameter count not enhanced! {enhanced_params} <= {original_params}")
        else:
            logging.info(f"NEAT: Parameters enhanced from {original_params} to {enhanced_params}")
        del baseline_model
    
    def forward(self, x):
        return self.enhanced_model(x)
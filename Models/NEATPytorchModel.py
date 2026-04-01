import logging
from Models.UniversalArchitectureModel import UniversalArchitectureModel
from Optimize.GraphMutator import FXMutator
import copy
import torch
import torch.nn as nn
import neat

class PyTorchNEATNetwork(nn.Module):
    """
    A native PyTorch module that perfectly mirrors a neat-python generated
    FeedForwardNetwork structure. It translates the graph of nodes and connections
    into registered PyTorch Parameters (weights and biases) allowing it to be
    differentiable, trainable, and compatible with torch.fx graph tracing.
    """
    def __init__(self, genome, config, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.node_evals = net.node_evals
        
        self.node_biases = nn.ParameterDict()
        self.node_responses = nn.ParameterDict()
        self.node_weights = nn.ParameterDict()
        self.activation_names = {}
        
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            self.node_biases[str(node)] = nn.Parameter(torch.tensor(float(bias)))
            self.node_responses[str(node)] = nn.Parameter(torch.tensor(float(response)))
            for in_node, weight in links:
                self.node_weights[f"{node}_{in_node}"] = nn.Parameter(torch.tensor(float(weight)))
            
            act_name = act_func.__name__ if hasattr(act_func, '__name__') else str(act_func)
            self.activation_names[str(node)] = act_name.lower()
            
    def forward(self, x):
        node_values = {}
        for i in range(self.input_size):
            node_values[-i - 1] = x[:, i]
            
        for node, _, _, _, _, links in self.node_evals:
            node_str = str(node)
            
            if len(links) == 0:
                node_val = torch.zeros(x.shape[0], device=x.device)
            else:
                node_val = None
                for in_node, _ in links:
                    w = self.node_weights[f"{node}_{in_node}"]
                    in_val = node_values[in_node]
                    step_val = in_val * w
                    if node_val is None:
                        node_val = step_val
                    else:
                        node_val = node_val + step_val
            
            node_val = node_val * self.node_responses[node_str] + self.node_biases[node_str]
            
            act_name = self.activation_names[node_str]
            if 'relu' in act_name:
                node_val = torch.relu(node_val)
            elif 'sigmoid' in act_name:
                node_val = torch.sigmoid(node_val)
            elif 'tanh' in act_name:
                node_val = torch.tanh(node_val)
            elif 'identity' in act_name:
                pass
            elif 'leaky_relu' in act_name:
                node_val = torch.nn.functional.leaky_relu(node_val)
            elif 'exp' in act_name:
                node_val = torch.exp(node_val)
            elif 'abs' in act_name:
                node_val = torch.abs(node_val)
            elif 'sin' in act_name:
                node_val = torch.sin(node_val)
            elif 'cos' in act_name:
                node_val = torch.cos(node_val)
            elif 'softplus' in act_name:
                node_val = torch.nn.functional.softplus(node_val)
            else:
                node_val = torch.sigmoid(node_val)
                
            node_values[node] = node_val
            
        outputs = []
        for i in range(self.output_size):
            if i in node_values:
                outputs.append(node_values[i])
            else:
                # Output IDs are 0, 1, 2... but in PyTorch/ONNX if it's completely disconnected, 
                # we just output zeros of the correct batch shape.
                # We use torch.zeros_like to keep it friendly with ONNX trace constraints.
                shape_tensor = x[:, 0]
                outputs.append(torch.zeros_like(shape_tensor, device=x.device))
        
        return torch.stack(outputs, dim=1)

class NEATPytorchModel(nn.Module):
    def __init__(self, genome, config, input_size, output_size, base_model_info, is_pytorch=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.base_model_info = base_model_info
        self.is_pytorch = is_pytorch
        
        # We no longer rely on neat.nn.FeedForwardNetwork during runtime training,
        # but we preserve it if external tools expect to read `.network`
        self.network = neat.nn.FeedForwardNetwork.create(genome, config)
        
        self._build_enhanced_universal_architecture(genome, config)
        
    def _build_enhanced_universal_architecture(self, genome, config):
        """Translate the NEAT genome's exact topological graph into PyTorch params"""
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
        
        logging.info(f"NEAT Native Topological Translation | "
                    f"Connections: {actual_connections} | Nodes: {total_nodes}")

        # Construct the native PyTorch structure that matches the found topology
        self.enhanced_model = PyTorchNEATNetwork(genome, config, self.input_size, self.output_size)

        enhanced_params = sum(p.numel() for p in self.enhanced_model.parameters())
        logging.info(f"NEAT: Native graph initialized with {enhanced_params} trainable parameters.")
    
    def forward(self, x):
        return self.enhanced_model(x)
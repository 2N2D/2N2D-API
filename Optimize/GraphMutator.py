import torch
import torch.nn as nn
import torch.fx as fx
import copy
import logging

class FXMutator:
    @staticmethod
    def apply_mutations(model: nn.Module, mutations: list) -> nn.Module:
        """
        Takes a list of mutation string commands and applies them sequentially.
        Examples: 'scale_width_1.5', 'add_layer', 'remove_layer', 'add_skip_connection'
        """
        cloned_model = copy.deepcopy(model)
        for mut in mutations:
            if mut.startswith('scale_width_'):
                factor = float(mut.split('_')[-1])
                cloned_model = FXMutator.mutate_width(cloned_model, factor)
            elif mut == 'add_layer':
                cloned_model = FXMutator.mutate_depth(cloned_model, 1)
            elif mut == 'remove_layer':
                cloned_model = FXMutator.mutate_depth(cloned_model, -1)
            elif mut == 'add_skip_connection':
                cloned_model = FXMutator.add_skip_connection(cloned_model)
        return cloned_model

    @staticmethod
    def mutate_width(model: nn.Module, scale_factor: float) -> nn.Module:
        """
        Symbolically traces the model and scales the width (out_features) 
        of internal Linear/Conv layers, propagating the dimension changes down the graph.
        """
        # Deepcopy to avoid modifying the original model instance parameters
        cloned_model = copy.deepcopy(model)
        
        if isinstance(cloned_model, fx.GraphModule):
            traced = cloned_model
        else:
            try:
                traced = fx.symbolic_trace(cloned_model)
            except Exception as e:
                logging.error(f"Failed to symbolically trace model for width mutation: {e}")
                return cloned_model

        modules = dict(traced.named_modules())
        
        # Maps node.name -> new output dimension size
        dim_overrides = {}

        for node in traced.graph.nodes:
            if node.op == 'call_module':
                target_mod = modules[node.target]
                
                if isinstance(target_mod, nn.Linear):
                    # Calculate new input features based on upstream overrides
                    new_in_features = target_mod.in_features
                    if len(node.args) > 0 and node.args[0].name in dim_overrides:
                        new_in_features = dim_overrides[node.args[0].name]
                    
                    # Scale output features if it's not the final output node
                    # (Heuristic: if it's the last linear before output, don't scale out_features to preserve target size)
                    users = list(node.users.keys())
                    # Check if this node eventually leads to output without another Linear
                    def reaches_output(n, visited=None):
                        if visited is None:
                            visited = set()
                        if n in visited:
                            return False
                        visited.add(n)
                        if n.op == 'output': return True
                        if n.op == 'call_module' and isinstance(modules.get(n.target), nn.Linear): return False
                        return any(reaches_output(u, visited) for u in n.users)
                    
                    is_output_edge = any(reaches_output(u) for u in users) if users else True
                    
                    if is_output_edge:
                        new_out_features = target_mod.out_features
                    else:
                        new_out_features = max(1, int(target_mod.out_features * scale_factor))
                    
                    # Update module
                    new_mod = nn.Linear(new_in_features, new_out_features)
                    # We initialize the weights identically, but truncated/padded
                    # For simplicity in this mutator, we just create a fresh initialized layer
                    setattr(traced, node.target, new_mod)
                    
                    # Record the new output dimension
                    dim_overrides[node.name] = new_out_features
                
                elif isinstance(target_mod, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.GELU)):
                    # Activations preserve the dimension of their input
                    if len(node.args) > 0 and node.args[0].name in dim_overrides:
                        dim_overrides[node.name] = dim_overrides[node.args[0].name]
                        
            elif node.op in ('call_function', 'call_method'):
                # Many operations preserve dimension or combine it (e.g. torch.add)
                # Handle basic propagation for the first argument:
                if getattr(node.target, '__name__', '') == 'cat':
                    # Special dimensionality combining for concat
                    # Try to infer based on inputs list if dims exist
                    if len(node.args) > 0 and isinstance(node.args[0], (list, tuple)):
                        sum_dim = 0
                        has_dims = False
                        for arg_node in node.args[0]:
                            if hasattr(arg_node, 'name') and arg_node.name in dim_overrides:
                                sum_dim += dim_overrides[arg_node.name]
                                has_dims = True
                        if has_dims:
                            dim_overrides[node.name] = sum_dim
                elif len(node.args) > 0 and hasattr(node.args[0], 'name') and node.args[0].name in dim_overrides:
                    dim_overrides[node.name] = dim_overrides[node.args[0].name]
        return traced

    @staticmethod
    def add_skip_connection(model: nn.Module) -> nn.Module:
        """
        Symbolically traces the model and attempts to add a skip connection
        (residual addition) between two compatible linear nodes.
        """
        cloned_model = copy.deepcopy(model)
        
        if isinstance(cloned_model, fx.GraphModule):
            traced = cloned_model
        else:
            try:
                traced = fx.symbolic_trace(cloned_model)
            except Exception as e:
                logging.error(f"Failed to trace model for skip connection: {e}")
                return cloned_model

        modules = dict(traced.named_modules())
        linear_nodes = [n for n in traced.graph.nodes if n.op == 'call_module' and isinstance(modules[n.target], nn.Linear)]
        
        # Need at least 3 linear nodes to have a meaningful skip connection over a block
        if len(linear_nodes) < 3:
            return traced

        # Try to find a pair with matching features
        for start_idx in range(len(linear_nodes) - 2):
            for end_idx in range(start_idx + 2, len(linear_nodes)):
                n_start = linear_nodes[start_idx]
                n_end = linear_nodes[end_idx]
                
                mod_start = modules[n_start.target]
                mod_end = modules[n_end.target]
                
                # Check for dimensionality match
                if mod_start.out_features == mod_end.out_features:
                    # Insert an addition operation right after the end node
                    with traced.graph.inserting_after(n_end):
                        # n_end's output + n_start's output
                        add_node = traced.graph.call_function(torch.add, (n_end, n_start))
                        
                        # Replace only downstream uses of n_end to use add_node instead
                        for user in list(n_end.users):
                            if user is not add_node:
                                user.replace_input_with(n_end, add_node)
                                
                    traced.recompile()
                    return traced

        return traced

    @staticmethod
    def mutate_depth(model: nn.Module, depth_change: int) -> nn.Module:
        """
        Symbolically traces the model and adds or removes sequential linear layers 
        to change the depth of the graph safely.
        """
        cloned_model = copy.deepcopy(model)
        
        if isinstance(cloned_model, fx.GraphModule):
            traced = cloned_model
        else:
            try:
                traced = fx.symbolic_trace(cloned_model)
            except Exception as e:
                logging.error(f"Failed to symbolically trace model for depth mutation: {e}")
                return cloned_model

        if depth_change == 0:
            return traced

        modules = dict(traced.named_modules())
        
        # Simple heuristic: find the first consecutive block of Linear layers
        linear_nodes = [n for n in traced.graph.nodes if n.op == 'call_module' and isinstance(modules[n.target], nn.Linear)]
        
        if not linear_nodes:
            return traced

        if depth_change > 0:
            # Insert a new layer after the first hidden linear layer
            target_node = linear_nodes[0]
            target_mod = modules[target_node.target]
            
            # Create a new module maintaining the same dimension
            new_mod_name = f"{target_node.target}_inserted_depth"
            new_mod = nn.Linear(target_mod.out_features, target_mod.out_features)
            # Add ReLU activation
            relu_mod_name = f"{target_node.target}_inserted_relu"
            traced.add_module(new_mod_name, new_mod)
            traced.add_module(relu_mod_name, nn.ReLU())
            
            # collect downstream users BEFORE inserting new nodes
            original_users = list(target_node.users.keys())
            
            with traced.graph.inserting_after(target_node):
                new_node = traced.graph.call_module(new_mod_name, args=(target_node,))
            with traced.graph.inserting_after(new_node):
                relu_node = traced.graph.call_module(relu_mod_name, args=(new_node,))
                
            # now redirect only the original downstream users
            for user in original_users:
                user.replace_input_with(target_node, relu_node)
                
        elif depth_change < 0 and len(linear_nodes) > 2:
            # Remove an intermediate linear layer
            node_to_remove = linear_nodes[1]
            mod_to_remove = modules[node_to_remove.target]
            
            # Find the input to this node
            input_node = node_to_remove.args[0]
            
            if mod_to_remove.in_features == mod_to_remove.out_features:
                node_to_remove.replace_all_uses_with(input_node)
                traced.graph.erase_node(node_to_remove)
            else:
                # Add a projection to reconcile dimensions instead of removing
                proj_name = f"proj_{node_to_remove.target}"
                traced.add_module(proj_name, nn.Linear(mod_to_remove.in_features, mod_to_remove.out_features))
                
                # The projection bridges the gap from input_node's size (in_features)
                # to what downstream nodes expect (out_features)
                with traced.graph.inserting_before(node_to_remove):
                    proj_node = traced.graph.call_module(proj_name, args=(input_node,))
                node_to_remove.replace_all_uses_with(proj_node)
                traced.graph.erase_node(node_to_remove)

        traced.recompile()
        return traced

    @staticmethod
    def extract_fx_blueprint(model: nn.Module) -> dict:
        """
        Extracts structural JSON metadata from an FX traced model to send to the frontend/UI.
        """
        if isinstance(model, fx.GraphModule):
            traced = model
        else:
            try:
                traced = fx.symbolic_trace(model)
            except Exception as e:
                logging.warning(f"Could not trace model for blueprint: {e}")
                return {"error": f"Could not trace model: {e}"}
            
        modules = dict(traced.named_modules())
        op_counts = {}
        layer_sizes = []
        input_size, output_size = None, None
        
        for node in traced.graph.nodes:
            if node.op == 'call_module':
                mod = modules[node.target]
                op_name = mod.__class__.__name__
                op_counts[op_name] = op_counts.get(op_name, 0) + 1
                
                if isinstance(mod, nn.Linear):
                    if input_size is None:
                        input_size = mod.in_features
                    output_size = mod.out_features
                    layer_sizes.append(mod.out_features)
                elif isinstance(mod, (nn.LSTM, nn.GRU, nn.RNN)):
                    if input_size is None:
                        input_size = getattr(mod, 'input_size', 0)
                    output_size = getattr(mod, 'hidden_size', 0)
                    layer_sizes.append(output_size)
                elif isinstance(mod, (nn.Conv1d, nn.Conv2d)):
                    if input_size is None:
                        input_size = getattr(mod, 'in_channels', 0)
                    output_size = getattr(mod, 'out_channels', 0)
                    layer_sizes.append(output_size)

        has_rnn = any(k in op_counts for k in ['LSTM', 'GRU', 'RNN'])
        has_conv = any(k in op_counts for k in ['Conv1d', 'Conv2d'])
        
        if has_rnn:
            model_type = 'lstm'
        elif has_conv:
            model_type = 'conv'
        elif 'Linear' in op_counts:
            model_type = 'feedforward'
        else:
            model_type = 'custom'
            
        return {
            'model_type': model_type,
            'layers': sum(op_counts.get(k, 0) for k in ['Linear', 'LSTM', 'GRU', 'RNN', 'Conv1d', 'Conv2d']),
            'neurons': max(layer_sizes) if layer_sizes else 0,
            'input_size': int(input_size) if input_size is not None else 0,
            'output_size': int(output_size) if output_size is not None else 0,
            'op_counts': op_counts,
            'layer_sizes': layer_sizes
        }

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

def create_neat_config(input_size, output_size):
    """Create a NEAT configuration file for the given problem"""
    config_path = "neat_config.ini"
    
    config_content = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000
pop_size              = 20
reset_on_extinction   = False

[DefaultGenome]
num_inputs            = {input_size}
num_outputs           = {output_size}
num_hidden            = 0
feed_forward          = True
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1
enabled_default         = True
enabled_mutate_rate     = 0.01
delay_init_mean         = 1.0
delay_init_stdev        = 0.0
delay_max_value         = 1.0
delay_min_value         = 0.0
delay_mutate_power      = 0.0
delay_mutate_rate       = 0.0
delay_replace_rate      = 0.0
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
node_add_prob           = 0.2
node_delete_prob        = 0.2

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    return config_path
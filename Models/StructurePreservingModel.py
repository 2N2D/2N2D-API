import torch.nn as nn
import logging
from Models.UniversalArchitectureModel import UniversalArchitectureModel

class StructurePreservingModel(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, num_layers, output_size, 
                 actual_input_size=None, actual_output_size=None, base_model_info=None):
        super(StructurePreservingModel, self).__init__()
        
        if model_type == 'lstm' or (base_model_info and base_model_info.get('has_lstm', False)):
            architecture_info = {
                'input_size': input_size,
                'output_size': output_size,
                'num_layers': num_layers,
                'max_width': hidden_size,
                'layer_sizes': [hidden_size] * num_layers,
                'has_lstm': True,
                'has_conv': False,
                'activation_type': 'relu',
                'lstm_hidden_size': hidden_size,
                'lstm_layers': num_layers,
                'post_lstm_layers': 1,
                'lstm_bidirectional': False,
                'lstm_return_sequences': False,
            }
        elif base_model_info is not None and base_model_info.get('has_conv', False):
            architecture_info = base_model_info.copy()
            architecture_info.update({
                'input_size': input_size,
                'output_size': output_size,
                'num_layers': num_layers,
                'max_width': hidden_size,
                'layer_sizes': [hidden_size] * num_layers,
            })
        else:
            architecture_info = {
                'input_size': input_size,
                'output_size': output_size,
                'num_layers': num_layers,
                'max_width': hidden_size,
                'layer_sizes': [hidden_size] * num_layers,
                'has_lstm': False,
                'has_conv': False,
                'activation_type': 'relu',
            }
        
        self.universal_model = UniversalArchitectureModel(
            architecture_info=architecture_info,
            enhancement_factor=1.0,
            actual_input_size=actual_input_size,
            actual_output_size=actual_output_size
        )
        
        if architecture_info.get('has_conv', False):
            conv_type = '2d' if architecture_info.get('has_conv2d', False) else '1d'
            logging.info(f"StructurePreservingModel: Preserving {conv_type} conv architecture with {len(architecture_info.get('conv_channels', []))} conv layers")
        elif architecture_info.get('has_lstm', False):
            logging.info(f"StructurePreservingModel: Preserving LSTM architecture")
        else:
            logging.info(f"StructurePreservingModel: Using feedforward architecture")
        
        self.needs_input_adapter = actual_input_size is not None and actual_input_size != input_size
        self.input_size = actual_input_size if self.needs_input_adapter else input_size
        self.model_type = model_type
        self.needs_output_adapter = actual_output_size is not None and actual_output_size != output_size
    
    def forward(self, x):
        return self.universal_model(x)
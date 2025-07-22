import logging
import torch.nn as nn
import torch
import numpy as np

class TransposeLayer(nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.perm = perm
    
    def forward(self, x):
        return x.permute(*self.perm)

class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        batch_size = x.size(0)
        new_shape = [batch_size] + [s if s != -1 else -1 for s in self.shape[1:]]
        return x.reshape(*new_shape)



class UniversalArchitectureModel(nn.Module):
    """Universal PyTorch model that can replicate and enhance any ONNX architecture"""
    def __init__(self, architecture_info, enhancement_factor=1.0, actual_input_size=None, actual_output_size=None):
        super().__init__()
        self.architecture_info = architecture_info
        self.enhancement_factor = max(1.0, enhancement_factor)  
        self.actual_input_size = actual_input_size or architecture_info['input_size']
        self.actual_output_size = actual_output_size or architecture_info['output_size']
        
        self.target_output_size = self.actual_output_size
        
        self.modified_architecture_info = architecture_info.copy()
        self.modified_architecture_info['output_size'] = self.target_output_size
        
        self.needs_input_adapter = actual_input_size is not None and actual_input_size != architecture_info['input_size']
        self.needs_output_adapter = False
        
        if self.needs_input_adapter:
            self.input_flatten = nn.Flatten()
            self.input_adapter = nn.Linear(actual_input_size, architecture_info['input_size'])
            self.original_input_shape = None
            
        self._build_universal_architecture()
    
    def _build_universal_architecture(self):
        """Build PyTorch architecture from universal ONNX analysis"""
        arch_info = self.modified_architecture_info
        layers = nn.ModuleList()
        
        enhancement_pattern = arch_info.get('enhancement_pattern', 0)
        
        logging.info(f"Building Universal Architecture:")
        logging.info(f"  Base model type: {arch_info.get('model_type', 'feedforward')}")
        logging.info(f"  Enhancement factor: {self.enhancement_factor}")
        logging.info(f"  Enhancement pattern: {enhancement_pattern}")
        logging.info(f"  Input size: {arch_info['input_size']}")
        logging.info(f"  ORIGINAL output size: {self.architecture_info['output_size']}")
        logging.info(f"  TARGET output size: {arch_info['output_size']} (USING THIS FOR MODEL CONSTRUCTION)")
        
        if self.enhancement_factor > 1.0:
            enhanced_layers = max(1, int(arch_info.get('num_layers', 1) * self.enhancement_factor))
            enhanced_width = max(arch_info['input_size'], int(arch_info.get('max_width', 64) * self.enhancement_factor))
            if enhancement_pattern == 1:  
                enhanced_width = int(enhanced_width * 1.2)
                logging.info(f"  Pattern 1: Width-focused enhancement -> {enhanced_width}")
            elif enhancement_pattern == 2:  
                enhanced_layers = max(enhanced_layers, enhanced_layers + 1)
                logging.info(f"  Pattern 2: Depth-focused enhancement -> {enhanced_layers} layers")
            elif enhancement_pattern == 3:  
                enhanced_width = int(enhanced_width * 1.1)
                enhanced_layers = max(enhanced_layers, enhanced_layers + 1)
                logging.info(f"  Pattern 3: Balanced enhancement -> {enhanced_layers} layers, {enhanced_width} width")
        else:
            enhanced_layers = arch_info.get('num_layers', 1)
            enhanced_width = arch_info.get('max_width', 64)
        
        logging.info(f"  Final architecture: {enhanced_layers} layers, {enhanced_width} max width")
        
        
        self._enhanced_layers = enhanced_layers
        self._enhanced_width = enhanced_width
        
        current_size = arch_info['input_size']       
        if arch_info.get('has_lstm', False):
            if self.enhancement_factor > 1.0:
                lstm_hidden = max(arch_info.get('lstm_hidden_size', 64), int(arch_info.get('lstm_hidden_size', 64) * self.enhancement_factor))
                lstm_layers = max(arch_info.get('lstm_layers', 1), int(arch_info.get('lstm_layers', 1) * self.enhancement_factor))
            else:
                lstm_hidden = arch_info.get('lstm_hidden_size', 64)
                lstm_layers = arch_info.get('lstm_layers', 1)
            
            logging.info(f"Building LSTM stack: hidden_size={lstm_hidden}, num_layers={lstm_layers}, enhancement={self.enhancement_factor}")
            self.lstm_layers = nn.ModuleList()
            layer_input_size = current_size
            
            for i in range(lstm_layers):
                lstm_layer = nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=lstm_hidden,
                    num_layers=1, 
                    batch_first=True,
                    dropout=0.0, 
                    bidirectional=arch_info.get('lstm_bidirectional', False)
                )
                self.lstm_layers.append(lstm_layer)
                layer_input_size = lstm_hidden * (2 if arch_info.get('lstm_bidirectional', False) else 1)
                if i < lstm_layers - 1:
                    self.lstm_layers.append(nn.Dropout(0.1))
            
            lstm_output_size = lstm_hidden * (2 if arch_info.get('lstm_bidirectional', False) else 1)
            current_size = lstm_output_size
            self.lstm_return_sequences = arch_info.get('lstm_return_sequences', False)
            self.num_lstm_layers = lstm_layers
            post_lstm_layers = arch_info.get('post_lstm_layers', 0)
            if post_lstm_layers > 0:
                if self.enhancement_factor > 1.0:
                    post_layers = max(1, int(post_lstm_layers * self.enhancement_factor))
                else:
                    post_layers = post_lstm_layers
                
                for i in range(post_layers):
                    if i == post_layers - 1:
                        layers.append(nn.Linear(current_size, arch_info['output_size']))
                    else:
                        next_size = max(enhanced_width // (i + 2), arch_info['output_size'])
                        layers.append(nn.Linear(current_size, next_size))
                        layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                        current_size = next_size
            else:
                layers.append(nn.Linear(current_size, arch_info['output_size']))
        elif arch_info.get('has_conv', False):
            conv_channels = arch_info.get('conv_channels', [32, 64])
            conv_type = '2d' if arch_info.get('has_conv2d', False) else '1d'  
            
            
            if self.enhancement_factor > 1.0:
                enhanced_channels = [max(c, int(c * self.enhancement_factor)) for c in conv_channels]
                if self.enhancement_factor > 1.5:
                   
                    extra_layer = max(enhanced_channels[-1], int(enhanced_channels[-1] * 1.2))
                    enhanced_channels.append(extra_layer)
            else:
                enhanced_channels = conv_channels[:]
            
            input_channels = arch_info.get('input_channels', 1)
            if input_channels <= 0:
                input_channels = 1
            

            conv_layers = []
            prev_channels = input_channels
            conv_details = arch_info.get('conv_details', [])
            pooling_details = arch_info.get('pooling_details', [])
            
            for i, channels in enumerate(enhanced_channels):
                conv_info = conv_details[i] if i < len(conv_details) else {}
                kernel_size = conv_info.get('kernel_size', [3])
                stride = conv_info.get('strides', [1])
                padding = self._calculate_padding(conv_info.get('pads', [0, 0, 0, 0]))
                dilations = conv_info.get('dilations', [1])
                
                if conv_type == '2d':
                    kernel_size = kernel_size if len(kernel_size) == 2 else [kernel_size[0], kernel_size[0]]
                    stride = stride if len(stride) == 2 else [stride[0], stride[0]]
                    dilation = dilations if len(dilations) == 2 else [dilations[0], dilations[0]]
                    
                    conv_layers.append(nn.Conv2d(
                        prev_channels, channels,
                        kernel_size=tuple(kernel_size),
                        stride=tuple(stride),
                        padding=padding,
                        dilation=tuple(dilation)
                    ))
                    
                    if arch_info.get('has_normalization', False):
                        conv_layers.append(nn.BatchNorm2d(channels))
                    
                    conv_layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                    
                    if i < len(pooling_details):
                        pool_info = pooling_details[i]
                        pool_kernel = pool_info.get('kernel_size', [2])
                        pool_stride = pool_info.get('strides', [2])
                        pool_kernel = pool_kernel if len(pool_kernel) == 2 else [pool_kernel[0], pool_kernel[0]]
                        pool_stride = pool_stride if len(pool_stride) == 2 else [pool_stride[0], pool_stride[0]]
                        
                        if pool_info.get('type') == 'MaxPool':
                            conv_layers.append(nn.MaxPool2d(
                                kernel_size=tuple(pool_kernel),
                                stride=tuple(pool_stride)
                            ))
                        elif pool_info.get('type') == 'AveragePool':
                            conv_layers.append(nn.AvgPool2d(
                                kernel_size=tuple(pool_kernel),
                                stride=tuple(pool_stride)
                            ))
                    
                else:
                    kernel_size = kernel_size[0] if kernel_size else 3
                    stride = stride[0] if stride else 1
                    dilation = dilations[0] if dilations else 1
                    
                    conv_layers.append(nn.Conv1d(
                        prev_channels, channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation
                    ))
                    
                    if arch_info.get('has_normalization', False):
                        conv_layers.append(nn.BatchNorm1d(channels))
                    
                    conv_layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                    
                    if i < len(pooling_details):
                        pool_info = pooling_details[i]
                        pool_kernel = pool_info.get('kernel_size', [2])[0]
                        pool_stride = pool_info.get('strides', [2])[0]
                        
                        if pool_info.get('type') == 'MaxPool':
                            conv_layers.append(nn.MaxPool1d(
                                kernel_size=pool_kernel,
                                stride=pool_stride
                            ))
                        elif pool_info.get('type') == 'AveragePool':
                            conv_layers.append(nn.AvgPool1d(
                                kernel_size=pool_kernel,
                                stride=pool_stride
                            ))
                
                prev_channels = channels
            
            layers.extend(conv_layers)
            
            structural_ops = arch_info.get('structural_ops', [])
            activation_ops = arch_info.get('activation_ops', [])
            op_sequence = arch_info.get('op_sequence', [])
            
            conv_end_position = -1
            for i, op in enumerate(op_sequence):
                if op == 'Conv':
                    conv_end_position = i
            
            post_conv_ops = []
            if conv_end_position >= 0:
                for struct_op in structural_ops:
                    if struct_op.get('position', 0) > conv_end_position:
                        post_conv_ops.append(struct_op)
                
                for act_op in activation_ops:
                    if act_op.get('position', 0) > conv_end_position:
                        post_conv_ops.append(act_op)
                
                post_conv_ops.sort(key=lambda x: x.get('position', 0))
                
                for op in post_conv_ops:
                    if op['type'] == 'Transpose':
                        perm = op.get('attributes', {}).get('perm', None)
                        if perm:
                            layers.append(TransposeLayer(perm))
                    elif op['type'] == 'Reshape':
                        shape = op.get('attributes', {}).get('shape', None)
                        if shape:
                            layers.append(ReshapeLayer(shape))
                    elif op['type'] == 'Flatten':
                        layers.append(nn.Flatten())
                    elif op['type'] == 'Softmax':
                        axis = op.get('attributes', {}).get('axis', -1)
                        layers.append(nn.Softmax(dim=axis))
                    elif op['type'] == 'LogSoftmax':
                        axis = op.get('attributes', {}).get('axis', -1)
                        layers.append(nn.LogSoftmax(dim=axis))
            
            if arch_info.get('post_conv_layers', 0) > 0:
                dense_sizes = arch_info.get('dense_layer_sizes', [enhanced_width])
                layers.append(nn.Flatten())
                current_size = None 
                
                for i, dense_size in enumerate(dense_sizes):
                    if self.enhancement_factor > 1.0:
                        enhanced_dense_size = max(dense_size, int(dense_size * self.enhancement_factor))
                    else:
                        enhanced_dense_size = dense_size
                    
                    if i == 0:
                        layers.append(nn.LazyLinear(enhanced_dense_size))
                    else:
                        layers.append(nn.Linear(current_size, enhanced_dense_size))
                    
                    layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                    
                    if self.enhancement_factor > 1.3:
                        layers.append(nn.Dropout(0.1))
                    
                    current_size = enhanced_dense_size
                
                layers.append(nn.Linear(current_size, arch_info['output_size']))
            else:
                layers.append(nn.Flatten())
                layers.append(nn.LazyLinear(arch_info['output_size']))
            
            self.conv_type = conv_type
            self.spatial_input_handling = True 
        
        else:
            layer_sizes = arch_info.get('layer_sizes', [enhanced_width] * enhanced_layers)
            if self.enhancement_factor > 1.0:
                enhanced_sizes = []
                for size in layer_sizes:
                    enhanced_size = max(size, int(size * self.enhancement_factor))
                    enhanced_sizes.append(enhanced_size)
                if self.enhancement_factor > 1.5:
                    extra_layers = int((self.enhancement_factor - 1.0) * 2)
                    for _ in range(extra_layers):                        enhanced_sizes.insert(-1, max(enhanced_sizes[-2] // 2, arch_info['output_size']))
            else:
                enhanced_sizes = layer_sizes[:]
                
            
            for i, size in enumerate(enhanced_sizes):
                if i == len(enhanced_sizes) - 1:
                    layers.append(nn.Linear(current_size, arch_info['output_size']))
                else:
                    layers.append(nn.Linear(current_size, size))
                    if arch_info.get('has_normalization', False):
                        layers.append(nn.BatchNorm1d(size))
                    
                    layers.append(self._get_activation(arch_info.get('activation_type', 'relu')))
                    if self.enhancement_factor > 1.2:
                        layers.append(nn.Dropout(0.1))
                    current_size = size
        
        self.layers = layers
        final_linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
        if final_linear_layers:
            final_layer = final_linear_layers[-1]
            expected_output = self.target_output_size
            actual_output = final_layer.out_features
            if actual_output != expected_output:
                logging.error(f"CRITICAL ERROR: Final layer outputs {actual_output} but target is {expected_output}!")
                raise ValueError(f"Final layer size mismatch: {actual_output} != {expected_output}")
            else:
                logging.info(f"✓ Final layer correctly outputs {actual_output} features to match target")
        
        self.arch_type = 'lstm' if arch_info.get('has_lstm', False) else 'conv' if arch_info.get('has_conv', False) else 'feedforward'
        
        logging.info(f"Universal architecture built: {self.arch_type} | Enhancement: {self.enhancement_factor:.2f} | Layers: {len([l for l in layers if isinstance(l, nn.Linear)])}")
        logging.info(f"  Model will output {self.target_output_size} features directly (no adapter bottleneck)")
        if self.arch_type == 'conv':
            conv_layer_count = len([l for l in layers if isinstance(l, (nn.Conv1d, nn.Conv2d))])
            has_lazy_linear = len([l for l in layers if isinstance(l, nn.LazyLinear)]) > 0
            logging.info(f"Conv architecture: {conv_layer_count} conv layers, Type: {getattr(self, 'conv_type', 'unknown')}, Spatial preserved: {has_lazy_linear}, No global pooling")
        elif self.arch_type == 'lstm':
            if hasattr(self, 'lstm_layers'):
                lstm_count = len([l for l in self.lstm_layers if isinstance(l, nn.LSTM)])
                dropout_count = len([l for l in self.lstm_layers if isinstance(l, nn.Dropout)])
                logging.info(f"LSTM architecture: {lstm_count} stacked LSTM layers with {dropout_count} dropout layers")
            elif hasattr(self, 'lstm'):
                logging.info(f"LSTM architecture: Single multi-layer LSTM (legacy)")
    
    def _get_activation(self, activation_type):
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leakyrelu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'silu': nn.SiLU(),  
            'elu': nn.ELU(),
            'prelu': nn.PReLU(),
            'relu6': nn.ReLU6(),
            'hardtanh': nn.Hardtanh(),
            'hardsigmoid': nn.Hardsigmoid(),
            'hardswish': nn.Hardswish(),
            'mish': nn.Mish(),  
        }
        return activations.get(activation_type.lower(), nn.ReLU())
    
    def _calculate_padding(self, pads):
        """Properly translate ONNX pads to PyTorch padding"""
        if len(pads) == 4:  
            pad_height = max(pads[0], pads[2])  
            pad_width = max(pads[1], pads[3])
            return (pad_height, pad_width)
        elif len(pads) == 2:  
            return max(pads[0], pads[1])
        else:
            return 0
    def forward(self, x):
        original_shape = x.shape
        if self.needs_input_adapter:
            if len(x.shape) > 2:
                self.original_input_shape = x.shape
                x = self.input_flatten(x)
            x = self.input_adapter(x)
        if self.arch_type == 'lstm':
            if hasattr(self, 'lstm_layers'):
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)
                elif len(x.shape) > 3:
                    batch_size = x.size(0)
                    x = x.view(batch_size, -1, x.size(-1))
                for layer in self.lstm_layers:
                    if isinstance(layer, nn.LSTM):
                        x, (h_n, c_n) = layer(x)
                    elif isinstance(layer, nn.Dropout):
                        x = layer(x)
                if hasattr(self, 'lstm_return_sequences') and self.lstm_return_sequences:
                    x = x.contiguous().view(x.size(0), -1)
                else:
                    if len(x.shape) == 3:
                        x = x[:, -1, :]  
                    else:
                        x = x
            elif hasattr(self, 'lstm'):
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)
                elif len(x.shape) > 3:
                    batch_size = x.size(0)
                    x = x.view(batch_size, -1, x.size(-1))
                
                lstm_out, (h_n, c_n) = self.lstm(x)
                if hasattr(self, 'lstm_return_sequences') and self.lstm_return_sequences:
                    x = lstm_out.contiguous().view(lstm_out.size(0), -1)
                else:
                    if len(lstm_out.shape) == 3:
                        x = lstm_out[:, -1, :]  
                    else:
                        x = lstm_out
                        
            for layer in self.layers:
                x = layer(x)
        
        elif self.arch_type == 'conv':
            conv_type = getattr(self, 'conv_type', '1d')
            if conv_type == '2d':
                if len(x.shape) == 2:
                    batch_size, features = x.shape
                    logging.warning(f"Converting tabular data ({features} features) to 2D spatial format. "
                                  f"This may not be semantically appropriate for non-spatial data.")
                    side_length = int(np.ceil(np.sqrt(features)))
                    target_features = side_length * side_length
                    
                    if target_features != features:
                        padding_size = target_features - features
                        if padding_size <= features:
                            padding = x[:, -padding_size:]
                        else:
                            padding = torch.zeros(batch_size, padding_size, device=x.device)
                        x = torch.cat([x, padding], dim=1)
                    input_channels = self.architecture_info.get('input_channels', 1)
                    x = x.view(batch_size, input_channels, side_length, side_length)
                    
                elif len(x.shape) == 3: 
                    x = x.unsqueeze(1)
                
                elif len(x.shape) == 4:
                    pass
                
            else:
                if len(x.shape) == 2: 
                    batch_size, features = x.shape
                    input_channels = self.architecture_info.get('input_channels', 1)
                    
                    if input_channels == 1:
                        x = x.unsqueeze(1)
                    else:
                        if features % input_channels == 0:
                            seq_length = features // input_channels
                            x = x.view(batch_size, input_channels, seq_length)
                        else:
                            logging.info(f"Features ({features}) not divisible by channels ({input_channels}), using single channel")
                            x = x.unsqueeze(1)
                
                elif len(x.shape) > 3:  
                    batch_size = x.size(0)
                    x = x.view(batch_size, 1, -1)
            for layer in self.layers:
                x = layer(x)
                
        else:
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            
            for layer in self.layers:
                x = layer(x)
        
        return x
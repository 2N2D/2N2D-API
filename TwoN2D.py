import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from Other.Data import (analyze_csv_data)
from Other.FileHandler import (createTempFile)

from Models.StructurePreservingModel import StructurePreservingModel
from Other.Utils import (get_device, make_json_serializable, create_neat_config)
from Optimize.Genetic import (optimize_with_genetic_deap)
from Optimize.Neat import (optimize_with_neat)

try:
    import onnxoptimizer
    HAS_ONNX_OPTIMIZER = True
except ImportError:
    HAS_ONNX_OPTIMIZER = False
    logging.warning("ONNX optimizer not available. Install with: pip install onnxoptimizer")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization_debug.log"),
        logging.StreamHandler()
    ]
)

def load_onnx_model(file_bytes):
    try:
        temp_file_path = createTempFile(file_bytes, ".onnx")

        model = onnx.load(temp_file_path)
        graph = model.graph

        nodes = []
        edges = []
        node_map = {}

        for i, node in enumerate(graph.node):
            label = node.name if node.name else node.op_type
            nodes.append({
                "id": i,
                "label": label,
                "title": str(node),
                "group": "operation",
            })
            if node.output:
                node_map[node.output[0]] = i

        for i, node in enumerate(graph.node):
            for input_name in node.input:
                if input_name in node_map:
                    edges.append({"from": node_map[input_name], "to": i})

        node_summary = []
        for node in graph.node:
            node_summary.append({
                "name": node.name if node.name else "Unnamed",
                "op_type": node.op_type
            })

        summary = {
            "ir_version": model.ir_version,
            "producer": model.producer_name,
            "inputs": [{"name": inp.name} for inp in graph.input],
            "outputs": [{"name": out.name} for out in graph.output],
            "nodes": node_summary,
            "node_count": len(graph.node)
        }

        os.remove(temp_file_path)

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": summary,
        }
    
    except Exception as e:
        logging.exception("Failed to load ONNX model")
        return {"error": str(e)}

def load_csv_data(binary_data, filename):
    try:
        data_io = io.BytesIO(binary_data)
        df = pd.read_csv(data_io)
        info = analyze_csv_data(df)


        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "filename": filename,
            "missing_values": df.isna().sum().to_dict(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns}
        }
        result = {"data": df.head(50).to_dict('records'), "summary": summary, "results": info}
        result = make_json_serializable(result)
        return result
    except Exception as e:
        logging.exception("Failed to load CSV data")
        return {"error": str(e)}

def analyze_onnx_model_universal(model_bytes):
    try:
        tempFilePath = createTempFile(model_bytes, ".onnx")
        model = onnx.load(tempFilePath)
        graph = model.graph
        input_shapes = {}
        for inp in graph.input:
            shape = []
            if hasattr(inp.type.tensor_type, 'shape'):
                for dim in inp.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value else None)
            input_shapes[inp.name] = shape
        
        output_shapes = {}
        for out in graph.output:
            shape = []
            if hasattr(out.type.tensor_type, 'shape'):
                for dim in out.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value else None)
            output_shapes[out.name] = shape
        op_counts = {}
        op_sequence = []
        layer_dimensions = []
        activation_types = set()
        
        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
            op_sequence.append(node.op_type)
            if node.op_type in ['Relu', 'Sigmoid', 'Tanh', 'LeakyRelu', 'Gelu', 'Swish']:
                activation_types.add(node.op_type.lower())
        weight_dimensions = {}
        for init in graph.initializer:
            weight_dimensions[init.name] = list(init.dims)
        has_lstm = op_counts.get('LSTM', 0) > 0
        has_gru = op_counts.get('GRU', 0) > 0
        has_rnn = op_counts.get('RNN', 0) > 0
        has_conv = op_counts.get('Conv', 0) > 0
        
        has_conv1d = False
        has_conv2d = False
        conv_channels = []
        input_channels = 1
        
        conv_details = []
        structural_ops = []
        activation_ops = []
        
        if has_conv:
            for node in graph.node:
                if node.op_type == 'Conv':
                    conv_info = {'kernel_size': [3], 'strides': [1], 'pads': [0, 0, 0, 0], 'dilations': [1]}
                    
                    for attr in node.attribute:
                        if attr.name == 'kernel_shape':
                            conv_info['kernel_size'] = list(attr.ints)
                        elif attr.name == 'strides':
                            conv_info['strides'] = list(attr.ints)
                        elif attr.name == 'pads':
                            conv_info['pads'] = list(attr.ints)
                        elif attr.name == 'dilations':
                            conv_info['dilations'] = list(attr.ints)
                    
                    for inp in node.input:
                        if inp in weight_dimensions:
                            dims = weight_dimensions[inp]
                            if len(dims) == 3:
                                has_conv1d = True
                                conv_info['out_channels'] = dims[0]
                                conv_info['in_channels'] = dims[1]
                                conv_info['type'] = '1d'
                                conv_channels.append(dims[0])
                                if len(conv_channels) == 1:
                                    input_channels = dims[1]
                            elif len(dims) == 4:
                                has_conv2d = True
                                conv_info['out_channels'] = dims[0]
                                conv_info['in_channels'] = dims[1]
                                conv_info['type'] = '2d'
                                conv_channels.append(dims[0])
                                if len(conv_channels) == 1:
                                    input_channels = dims[1]
                            break
                    
                    conv_details.append(conv_info)
            
            if not conv_channels:
                conv_channels = [32, 64]
        
        for i, op_type in enumerate(op_sequence):
            if op_type in ['Transpose', 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 
                          'Concat', 'Split', 'Slice', 'Gather', 'Expand']:
                for node in graph.node:
                    if node.op_type == op_type:
                        structural_info = {'type': op_type, 'position': i, 'attributes': {}}
                        
                        for attr in node.attribute:
                            if attr.name == 'perm' and op_type == 'Transpose':
                                structural_info['attributes']['perm'] = list(attr.ints)
                            elif attr.name == 'shape' and op_type == 'Reshape':
                                structural_info['attributes']['shape'] = list(attr.ints)
                            elif attr.name == 'axes':
                                structural_info['attributes']['axes'] = list(attr.ints)
                            elif attr.name == 'axis':
                                structural_info['attributes']['axis'] = attr.i
                            elif attr.name == 'starts':
                                structural_info['attributes']['starts'] = list(attr.ints)
                            elif attr.name == 'ends':
                                structural_info['attributes']['ends'] = list(attr.ints)
                        
                        structural_ops.append(structural_info)
                        break
            
            elif op_type in ['Softmax', 'LogSoftmax']:
                for node in graph.node:
                    if node.op_type == op_type:
                        activation_info = {'type': op_type, 'position': i, 'attributes': {}}
                        
                        for attr in node.attribute:
                            if attr.name == 'axis':
                                activation_info['attributes']['axis'] = attr.i
                            elif attr.name == 'alpha':
                                activation_info['attributes']['alpha'] = attr.f
                        
                        activation_ops.append(activation_info)
                        break
        
        has_attention = any(op in op_counts for op in ['Attention', 'MultiHeadAttention'])
        has_normalization = any(op in op_counts for op in ['BatchNormalization', 'LayerNormalization', 'InstanceNormalization'])
        has_pooling = any(op in op_counts for op in ['MaxPool', 'AveragePool', 'GlobalAveragePool', 'GlobalMaxPool'])
        
        pooling_details = []
        if has_pooling:
            for node in graph.node:
                if node.op_type in ['MaxPool', 'AveragePool']:
                    pool_info = {'type': node.op_type, 'kernel_size': [2], 'strides': [2], 'pads': [0, 0, 0, 0]}
                    
                    for attr in node.attribute:
                        if attr.name == 'kernel_shape':
                            pool_info['kernel_size'] = list(attr.ints)
                        elif attr.name == 'strides':
                            pool_info['strides'] = list(attr.ints)
                        elif attr.name == 'pads':
                            pool_info['pads'] = list(attr.ints)
                    
                    pooling_details.append(pool_info)
        
        linear_ops = ['Gemm', 'MatMul', 'Linear']
        linear_layers = sum(op_counts.get(op, 0) for op in linear_ops)
        input_size = 0
        output_size = 1
        max_width = 64
        layer_sizes = []
        if input_shapes:
            first_input = list(input_shapes.values())[0]
            if first_input and len(first_input) > 1:
                if first_input[-1] is not None and first_input[-1] > 0:
                    input_size = first_input[-1]
                elif len(first_input) > 1 and first_input[1] is not None and first_input[1] > 0:
                    input_size = first_input[1]
        if output_shapes:
            first_output = list(output_shapes.values())[0]
            if first_output and len(first_output) > 0:
                if first_output[-1] is not None and first_output[-1] > 0:
                    output_size = first_output[-1]
                elif len(first_output) > 1 and first_output[1] is not None and first_output[1] > 0:
                    output_size = first_output[1]
        linear_weights = []
        extracted_layer_sizes = []
        for node in graph.node:
            if node.op_type in linear_ops:
                for inp in node.input:
                    if inp in weight_dimensions:
                        dims = weight_dimensions[inp]
                        if len(dims) >= 2:
                            linear_weights.append(dims)
                            output_dim = dims[0]
                            input_dim = dims[1]
                            extracted_layer_sizes.append(output_dim)
                            max_width = max(max_width, output_dim, input_dim)
                            if input_size <= 0 and len(extracted_layer_sizes) == 1:
                                input_size = input_dim
        if extracted_layer_sizes:
            layer_sizes = extracted_layer_sizes[:-1] if len(extracted_layer_sizes) > 1 else [max_width]
            if extracted_layer_sizes and output_size == 1:
                output_size = extracted_layer_sizes[-1]
        lstm_hidden_size = 64
        lstm_layers = 1
        if has_lstm or has_gru or has_rnn:
            logging.info(f"Detected recurrent layers: LSTM={has_lstm}, GRU={has_gru}, RNN={has_rnn}")
            for node in graph.node:
                if node.op_type in ['LSTM', 'GRU', 'RNN']:
                    logging.info(f"Found {node.op_type} node: {node.name}")
                    for inp in node.input:
                        if inp in weight_dimensions:
                            dims = weight_dimensions[inp]
                            logging.info(f"  Weight dimensions: {dims}")
                            if len(dims) >= 2:
                                if node.op_type == 'LSTM':
                                    potential_hidden = dims[0] // 4
                                    if potential_hidden > 0:
                                        lstm_hidden_size = max(lstm_hidden_size, potential_hidden)
                                        logging.info(f"  LSTM hidden size calculated: {potential_hidden}")
                                else:
                                    lstm_hidden_size = max(lstm_hidden_size, dims[0])
                                    logging.info(f"  {node.op_type} hidden size: {dims[0]}")
                                break
                    for attr in node.attribute:
                        if attr.name == 'hidden_size':
                            lstm_hidden_size = max(lstm_hidden_size, attr.i)
                            logging.info(f"  Hidden size from attributes: {attr.i}")
                        elif attr.name == 'num_layers' or attr.name == 'direction':
                            if attr.name == 'num_layers':
                                lstm_layers = max(lstm_layers, attr.i)
                                logging.info(f"  Number of layers from attributes: {attr.i}")
            logging.info(f"Final LSTM config: hidden_size={lstm_hidden_size}, layers={lstm_layers}")
        post_conv_layers = 0
        if has_conv and linear_layers > 0:
            conv_found = False
            for op in op_sequence:
                if op == 'Conv':
                    conv_found = True
                elif conv_found and op in linear_ops:
                    post_conv_layers += 1
            post_conv_layers = max(0, post_conv_layers - 1)
        num_layers = max(1, len([op for op in op_sequence if op in linear_ops + ['LSTM', 'GRU', 'RNN', 'Conv']]))
        primary_activation = 'relu'
        if activation_types:
            activation_priority = ['gelu', 'swish', 'leakyrelu', 'relu', 'tanh', 'sigmoid']
            for act in activation_priority:
                if act in activation_types:
                    primary_activation = act
                    break
        post_recurrent_layers = 0
        if has_lstm or has_gru or has_rnn:
            recurrent_found = False
            for op in op_sequence:
                if op in ['LSTM', 'GRU', 'RNN']:
                    recurrent_found = True
                elif recurrent_found and op in linear_ops:
                    post_recurrent_layers += 1
        if input_size <= 0:
            input_size = max(layer_sizes[0] if layer_sizes else 64, 1)
        if not layer_sizes:
            layer_sizes = [max_width] * max(1, num_layers - 1)
        architecture_info = {
            'input_size': input_size,
            'output_size': output_size,
            'num_layers': num_layers,
            'max_width': max_width,
            'layer_sizes': layer_sizes,
            'has_lstm': has_lstm,
            'has_gru': has_gru,
            'has_rnn': has_rnn,
            'has_conv': has_conv,
            'has_conv1d': has_conv1d,
            'has_conv2d': has_conv2d,
            'has_attention': has_attention,
            'has_normalization': has_normalization,
            'activation_type': primary_activation,
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_layers': lstm_layers,
            'lstm_bidirectional': False,
            'lstm_return_sequences': False,
            'post_lstm_layers': post_recurrent_layers,
            'conv_channels': conv_channels if conv_channels else [32, 64],
            'input_channels': input_channels,
            'post_conv_layers': post_conv_layers,
            'conv_details': conv_details,
            'pooling_details': pooling_details,
            'structural_ops': structural_ops,
            'activation_ops': activation_ops,
            'has_pooling': has_pooling,
            'dense_layer_sizes': [max_width] if post_conv_layers > 0 else [],
            'op_counts': op_counts,
            'op_sequence': op_sequence,
            'model_type': 'lstm' if (has_lstm or has_gru or has_rnn) else 'conv' if has_conv else 'feedforward',
            'hidden_size': max_width,
            'is_recurrent': has_lstm or has_gru or has_rnn,
            'layers': num_layers,
            'neurons': max_width
        }
        
        logging.info(f"Universal ONNX Analysis: {architecture_info['model_type']} | "
                    f"Layers: {num_layers} | Max Width: {max_width} | "
                    f"Input: {input_size} | Output: {output_size}")
        if has_conv:
            conv_type_str = '2d' if has_conv2d else '1d'
            logging.info(f"Conv details: Type={conv_type_str}, Channels={conv_channels}, Input_channels={input_channels}, Post-conv layers={post_conv_layers}")
            logging.info(f"Conv channel progression: {input_channels} -> {' -> '.join(map(str, conv_channels))}")
            if conv_details:
                for i, detail in enumerate(conv_details):
                    kernel = detail.get('kernel_size', [3])
                    stride = detail.get('strides', [1])
                    logging.info(f"Conv layer {i}: channels={detail.get('out_channels', 'unknown')}, kernel={kernel}, stride={stride}")
            if pooling_details:
                for i, detail in enumerate(pooling_details):
                    logging.info(f"Pool layer {i}: type={detail.get('type', 'unknown')}, kernel={detail.get('kernel_size', [2])}")
            if structural_ops:
                struct_types = [op['type'] for op in structural_ops]
                logging.info(f"Structural ops: {', '.join(struct_types)}")
            if activation_ops:
                act_types = [op['type'] for op in activation_ops]
                logging.info(f"Special activations: {', '.join(act_types)}")
        logging.info(f"Operators: {dict(list(op_counts.items())[:5])}...")
        
        os.remove(tempFilePath)
        
        architecture_info = make_json_serializable(architecture_info)
        return architecture_info
        
    except Exception as e:
        logging.exception("Error analyzing ONNX model universally")
        if 'tempFilePath' in locals():
            try:
                os.remove(tempFilePath)
            except:
                pass
        return {
            'input_size': 64,
            'output_size': 1,
            'num_layers': 2,
            'max_width': 128,
            'layer_sizes': [128, 64],
            'has_lstm': False,
            'has_gru': False,
            'has_rnn': False,
            'has_conv': False,
            'has_conv1d': False,
            'has_conv2d': False,
            'has_attention': False,
            'has_normalization': False,
            'activation_type': 'relu',
            'lstm_hidden_size': 64,
            'lstm_layers': 1,
            'post_lstm_layers': 0,
            'conv_channels': [],
            'conv_details': [],
            'pooling_details': [],
            'structural_ops': [],
            'activation_ops': [],
            'has_pooling': False,
            'op_counts': {},
            'op_sequence': [],
            'model_type': 'feedforward',
            'hidden_size': 128,
            'is_recurrent': False,
            'layers': 1,
            'neurons': 64
        }

def analyze_onnx_model(model_bytes):
    return analyze_onnx_model_universal(model_bytes)

def validate_model_shapes(model, input_size, output_size, sample_input=None, device=None):
    """Validate that model can handle expected input/output shapes"""
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if sample_input is None:
            sample_input = torch.randn(1, input_size, device=device)
        
        
        model = model.to(device)
        sample_input = sample_input.to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        actual_output_size = output.shape[1] if len(output.shape) > 1 else 1
        if actual_output_size != output_size:
            logging.warning(f"Model output size {actual_output_size} doesn't match expected {output_size}")
            return False, f"Shape mismatch: expected {output_size}, got {actual_output_size}"
        return True, "Shape validation passed"
    except Exception as e:
        return False, f"Shape validation failed: {str(e)}"

def validate_onnx_export_shapes(model, input_size, output_size, device=None):
    """Validate shapes before ONNX export"""
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        dummy_input = torch.randn(1, input_size, device=device)
        model.eval()
        with torch.no_grad():
            test_output = model(dummy_input)
        if len(test_output.shape) != 2:
            return False, f"Expected 2D output, got {len(test_output.shape)}D"
        if test_output.shape[1] != output_size:
            return False, f"Output size mismatch: expected {output_size}, got {test_output.shape[1]}"
        temp_path = os.path.join(tempfile.gettempdir(), "test_export.onnx")
        try:
            
            model_cpu = model.to("cpu")
            dummy_input_cpu = dummy_input.to("cpu")
            
            torch.onnx.export(
                model_cpu,
                dummy_input_cpu,
                temp_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=12,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL
            )
            if os.path.exists(temp_path):
                test_model = onnx.load(temp_path)
                onnx.checker.check_model(test_model)
                os.remove(temp_path)
                return True, "ONNX export validation passed"
            else:
                return False, "ONNX export failed - no file created"
        except Exception as export_error:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False, f"ONNX export test failed: {str(export_error)}"
    except Exception as e:
        return False, f"Shape validation failed: {str(e)}"

def find_optimal_architecture(onnx_bytes, input_features, target_feature, df: pd.DataFrame, max_epochs=5, 
                            strategy='brute-force', generations=5, status_callback=None):
    """
    Enhanced version of the original function with new optimization strategies.
    Maintains backward compatibility while adding NEAT and DEAP genetic algorithms.
    """
    
    def send_status(message):
        if status_callback:
            status_callback(message)
        else:
            logging.info(message.get("status", str(message)))

    try:
        
        device = get_device()
        
        
        torch.set_num_threads(4)  
        
        if df is None:
            send_status({"status": "No data loaded", "error": True})
            return {"error": "No data loaded"}

        if onnx_bytes is None:
            send_status({"status": "No ONNX model loaded", "error": True})
            return {"error": "No ONNX model loaded"}

        send_status({"status": "Analyzing model...", "progress": 5})
        model_info = analyze_onnx_model(onnx_bytes)
        base_layers = model_info.get('layers', 1)
        base_neurons = model_info.get('neurons', 64)
        model_type = model_info.get('model_type', 'feedforward')
        input_size = model_info.get('input_size', 0)
        output_size = model_info.get('output_size', 1)

        send_status({
            "status": f"Base model: {model_type}, {base_layers} layers, {base_neurons} neurons",
            "progress": 15
        })  

        missing_values = df[input_features + [target_feature]].isna().sum()
        if missing_values.sum() > 0:
            logging.warning(f"Found missing values in data: {missing_values[missing_values > 0]}")
            logging.info("Filling missing values with column means")
            df[input_features + [target_feature]] = df[input_features + [target_feature]].fillna(df[input_features + [target_feature]].mean())
        
        X = df[input_features].values.astype(np.float32)
        y = df[target_feature].values.astype(np.float32).reshape(-1, 1)
        if np.isnan(X).any() or np.isinf(X).any():
            logging.warning("Found NaN or infinite values in features. Replacing with zeros.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(y).any() or np.isinf(y).any():
            logging.warning("Found NaN or infinite values in target. Replacing with zeros.")
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)
        logging.info(f"Target feature '{target_feature}' statistics:")
        logging.info(f"  Original range: [{y.min():.6f}, {y.max():.6f}]")
        logging.info(f"  Original std: {y.std():.6f}")
        logging.info(f"  Scaled range: [{y_train.min():.6f}, {y_train.max():.6f}]")
        logging.info(f"  Scaled std: {y_train.std():.6f}")
        
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)
        
        
        if device.type == "cuda":
            batch_size = min(128, len(X_train) // 4)  
            logging.info(f"Using GPU-optimized batch size: {batch_size}")
        else:
            batch_size = 32
            logging.info(f"Using CPU batch size: {batch_size}")
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        actual_input_size = X.shape[1]
        actual_output_size = y.shape[1]
        
        logging.info(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        logging.info(f"Feature shape: {X_train.shape}, Target shape: {y_train.shape}")
        logging.info(f"Actual input size: {actual_input_size}, Actual output size: {actual_output_size}")
        logging.info(f"Model expected input size: {input_size}, output size: {output_size}")
        logging.info(f"Using optimization strategy: {strategy}")
        send_status({"status": "Evaluating baseline performance for reference...", "progress": 10})
        baseline_model = StructurePreservingModel(
            model_type=model_type,
            input_size=input_size if input_size > 0 else actual_input_size,
            hidden_size=base_neurons,
            num_layers=base_layers,
            output_size=output_size if output_size > 0 else 1,
            actual_input_size=actual_input_size,
            actual_output_size=actual_output_size,
            base_model_info=model_info
        )
        baseline_model = baseline_model.to(device)  
        criterion = nn.MSELoss()
        optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
        
        for epoch in range(max_epochs):
            baseline_model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = baseline_model(batch_X)
                if outputs.shape != batch_y.shape:
                    if outputs.shape[1] > batch_y.shape[1]:
                        outputs = outputs[:, :batch_y.shape[1]]
                    elif outputs.shape[1] < batch_y.shape[1]:
                        last_col = outputs[:, -1:].repeat(1, batch_y.shape[1] - outputs.shape[1])
                        outputs = torch.cat([outputs, last_col], dim=1)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        baseline_model.eval()
        with torch.no_grad():
            baseline_outputs = []
            baseline_targets = []
            for batch_X, batch_y in test_loader:
                outputs = baseline_model(batch_X)
                if outputs.shape != batch_y.shape:
                    if outputs.shape[1] > batch_y.shape[1]:
                        outputs = outputs[:, :batch_y.shape[1]]
                    elif outputs.shape[1] < batch_y.shape[1]:
                        last_col = outputs[:, -1:].repeat(1, batch_y.shape[1] - outputs.shape[1])
                        outputs = torch.cat([outputs, last_col], dim=1)
                baseline_outputs.append(outputs.cpu().numpy())
                baseline_targets.append(batch_y.cpu().numpy())
            
            baseline_outputs = np.vstack(baseline_outputs)
            baseline_targets = np.vstack(baseline_targets)
            baseline_outputs_unscaled = scaler_y.inverse_transform(baseline_outputs)
            baseline_targets_unscaled = scaler_y.inverse_transform(baseline_targets)
            
            baseline_test_loss = mean_squared_error(baseline_targets_unscaled, baseline_outputs_unscaled)
            baseline_r2 = r2_score(baseline_targets_unscaled, baseline_outputs_unscaled)
        
        logging.info(f"✓ Baseline reference performance (unscaled): Loss={baseline_test_loss:.6f}, R2={baseline_r2:.6f}")
        send_status({
            "status": f"Reference baseline: Loss={baseline_test_loss:.4f}, R2={baseline_r2:.4f}",
            "progress": 15
        })

        best_loss = float('inf')
        best_result = None
        best_model = None
        results = []

        if strategy == 'brute-force':
            neurons_options = [max(16, base_neurons // 2), base_neurons, base_neurons * 2]  
            layers_options = [max(1, base_layers - 1), base_layers, base_layers + 1]       

            current_iteration = 0
            total_iterations = len(neurons_options) * len(layers_options)

            for hidden_size in neurons_options:
                for num_layers in layers_options:
                    current_iteration += 1
                    progress_percent = 20 + int((current_iteration / total_iterations) * 60)

                    send_status({
                        "status": f"Testing {num_layers} layers, {hidden_size} neurons",
                        "progress": progress_percent
                    })
                    
                    model = StructurePreservingModel(
                        model_type=model_type,
                        input_size=input_size if input_size > 0 else actual_input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        output_size=output_size if output_size > 0 else 1,
                        actual_input_size=actual_input_size,
                        actual_output_size=actual_output_size,
                        base_model_info=model_info 
                    )
                    model = model.to(device)  
                    
                    criterion = nn.MSELoss()
                    if model_type == 'lstm':
                        lr = 0.01  
                        actual_epochs = max(max_epochs * 2, 20)
                    elif model_type == 'conv':
                        lr = 0.001  
                        actual_epochs = max(max_epochs, 15)
                    else:
                        lr = 0.001  
                        actual_epochs = max_epochs  
                    
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
                    val_split_size = int(0.2 * len(X_train_tensor))
                    train_subset = X_train_tensor[val_split_size:]
                    val_subset = X_train_tensor[:val_split_size]
                    train_targets_subset = y_train_tensor[val_split_size:]
                    val_targets_subset = y_train_tensor[:val_split_size]
                    
                    train_subset_dataset = TensorDataset(train_subset, train_targets_subset)
                    train_subset_loader = DataLoader(train_subset_dataset, batch_size=batch_size, shuffle=True)
                    
                    val_subset_dataset = TensorDataset(val_subset, val_targets_subset)
                    val_subset_loader = DataLoader(val_subset_dataset, batch_size=batch_size, shuffle=False)
                    best_val_loss = float('inf')
                    patience_counter = 0
                    
                    for epoch in range(actual_epochs):
                        model.train()
                        epoch_loss = 0
                        batch_count = 0
                        
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            if outputs.shape != batch_y.shape:
                                if outputs.shape[0] == batch_y.shape[0]:  
                                    if outputs.shape[1] > batch_y.shape[1]:
                                        outputs = outputs[:, :batch_y.shape[1]]
                                    elif outputs.shape[1] < batch_y.shape[1]:
                                        last_col = outputs[:, -1:].repeat(1, batch_y.shape[1] - outputs.shape[1])
                                        outputs = torch.cat([outputs, last_col], dim=1)
                            
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            
                            optimizer.step()
                            epoch_loss += loss.item()
                            batch_count += 1
                        
                        avg_epoch_loss = epoch_loss / batch_count
                        model.eval()
                        val_loss = 0
                        val_batches = 0
                        with torch.no_grad():
                            for val_X, val_y in val_subset_loader:
                                val_X, val_y = val_X.to(device), val_y.to(device)
                                val_outputs = model(val_X)
                                if val_outputs.shape != val_y.shape:
                                    if val_outputs.shape[0] == val_y.shape[0]:  
                                        if val_outputs.shape[1] > val_y.shape[1]:
                                            val_outputs = val_outputs[:, :val_y.shape[1]]
                                        elif val_outputs.shape[1] < val_y.shape[1]:
                                            last_col = val_outputs[:, -1:].repeat(1, val_y.shape[1] - val_outputs.shape[1])
                                            val_outputs = torch.cat([val_outputs, last_col], dim=1)
                                
                                val_loss += criterion(val_outputs, val_y).item()
                                val_batches += 1
                        
                        avg_val_loss = val_loss / max(val_batches, 1)
                        scheduler.step(avg_val_loss)
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= 5:  
                                logging.info(f"Early stopping at epoch {epoch+1} - Val loss: {avg_val_loss:.6f}")
                                break
                    
                    model.eval()
                    with torch.no_grad():
                        all_outputs = []
                        all_targets = []
                        
                        for batch_X, batch_y in test_loader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            outputs = model(batch_X)
                            if outputs.shape != batch_y.shape:
                                if outputs.shape[0] == batch_y.shape[0]:  
                                    if outputs.shape[1] > batch_y.shape[1]:
                                        outputs = outputs[:, :batch_y.shape[1]]
                                    elif outputs.shape[1] < batch_y.shape[1]:
                                        last_col = outputs[:, -1:].repeat(1, batch_y.shape[1] - outputs.shape[1])
                                        outputs = torch.cat([outputs, last_col], dim=1)
                            
                            all_outputs.append(outputs.cpu().numpy())
                            all_targets.append(batch_y.cpu().numpy())
                        
                        all_outputs = np.vstack(all_outputs)
                        all_targets = np.vstack(all_targets)
                        if all_outputs.shape[1] != all_targets.shape[1]:
                            if all_outputs.shape[1] > all_targets.shape[1]:
                                all_outputs = all_outputs[:, :all_targets.shape[1]]
                            else:
                                all_targets = all_targets[:, :all_outputs.shape[1]]
                        all_outputs_unscaled = scaler_y.inverse_transform(all_outputs)
                        all_targets_unscaled = scaler_y.inverse_transform(all_targets)
                        
                        test_loss = mean_squared_error(all_targets_unscaled, all_outputs_unscaled)
                        r2 = r2_score(all_targets_unscaled, all_outputs_unscaled)
                    improvement_vs_best = (best_loss - test_loss) / best_loss * 100 if best_loss != float('inf') else 0
                    
                    logging.info(f"Model {current_iteration}/{total_iterations}: "
                               f"{num_layers}L/{hidden_size}N - "
                               f"Loss: {test_loss:.6f}, R2: {r2:.6f}, "
                               f"Current best: {best_loss:.6f}, "
                               f"{'✓' if test_loss < best_loss else '✗'} "
                               f"{improvement_vs_best:+.1f}%")

                    results.append({
                        "neurons": int(hidden_size), 
                        "layers": int(num_layers), 
                        "test_loss": float(test_loss), 
                        "r2_score": float(r2)
                    })
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_result = {
                            "neurons": int(hidden_size), 
                            "layers": int(num_layers), 
                            "test_loss": float(test_loss), 
                            "r2_score": float(r2)
                        }
                        best_model = model
                        send_status({
                            "status": f"New best: {num_layers}L/{hidden_size}N - Loss: {test_loss:.4f}",
                            "progress": progress_percent
                        })

            send_status({"status": "Optimization complete", "progress": 90})
            if best_model is None:
                logging.warning("No models were successfully trained during optimization.")
                best_model = StructurePreservingModel(
                    model_type=model_type,
                    input_size=input_size if input_size > 0 else actual_input_size,
                    hidden_size=base_neurons,
                    num_layers=base_layers,
                    output_size=output_size if output_size > 0 else 1,
                    actual_input_size=actual_input_size,
                    actual_output_size=actual_output_size,
                    base_model_info=model_info
                )
                best_result = {
                    "neurons": int(base_neurons), 
                    "layers": int(base_layers), 
                    "test_loss": float('inf'), 
                    "r2_score": float(0)
                }
                best_loss = float('inf')
            
            logging.info(f"✓ Optimization complete! Best model: {best_result['layers']}L/{best_result['neurons']}N - Loss: {best_loss:.6f}")
            send_status({
                "status": f"Optimization complete! Best: {best_result['layers']}L/{best_result['neurons']}N",
                "progress": 90
            })
            
        elif strategy == 'neat':
            send_status({"status": "Starting NEAT optimization...", "progress": 20})
            config_path = create_neat_config(actual_input_size, actual_output_size)
            best_model = optimize_with_neat(
                X_train, y_train, 
                actual_input_size, actual_output_size,
                config_path, model_info, max_epochs, send_status, device
            )
            send_status({"status": "Training best NEAT model with backpropagation...", "progress": 75})
            train_loader_neat = DataLoader(
                TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32, device=device),
                    torch.tensor(y_train, dtype=torch.float32, device=device)
                ), 
                batch_size=32, 
                shuffle=True
            )
            
            optimizer = optim.Adam(best_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            final_epochs = max_epochs*2
            for epoch in range(final_epochs):
                best_model.train()
                for batch_X, batch_y in train_loader_neat:
                    optimizer.zero_grad()
                    outputs = best_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                
                if epoch % max(1, final_epochs // 5) == 0:  
                    progress = 75 + int((epoch / final_epochs) * 10)  
                    send_status({"status": f"NEAT backpropagation training: epoch {epoch+1}/{final_epochs}", "progress": progress})
            
            send_status({"status": "Evaluating NEAT model performance...", "progress": 85})
            best_model = best_model.to(device)  
            best_model.eval()
            with torch.no_grad():
                all_outputs = []
                all_targets = []
                
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = best_model(batch_X)
                    all_outputs.append(outputs.cpu().numpy())
                    all_targets.append(batch_y.cpu().numpy())
                
                all_outputs = np.vstack(all_outputs)
                all_targets = np.vstack(all_targets)
                all_outputs_unscaled = scaler_y.inverse_transform(all_outputs)
                all_targets_unscaled = scaler_y.inverse_transform(all_targets)
                
                best_loss = mean_squared_error(all_targets_unscaled, all_outputs_unscaled)
                r2 = r2_score(all_targets_unscaled, all_outputs_unscaled)
            
            
            neat_arch = getattr(best_model, '_neat_architecture', {})
            layers = neat_arch.get('layers', model_info.get('num_layers', 1))
            neurons = neat_arch.get('neurons', model_info.get('max_width', 64))
            
            best_result = {
                "strategy": "NEAT", 
                "test_loss": float(best_loss), 
                "r2_score": float(r2), 
                "layers": int(layers), 
                "neurons": int(neurons),
                "neat_nodes": neat_arch.get('neat_nodes', 0),
                "neat_connections": neat_arch.get('neat_connections', 0),
                "enhancement_factor": neat_arch.get('enhancement_factor', 1.0),
                "base_model": model_info
            }
            results = [{
                "strategy": "NEAT", 
                "generations": max_epochs, 
                "test_loss": float(best_loss), 
                "r2_score": float(r2),
                "layers": int(layers),
                "neurons": int(neurons)
            }]
            
        elif strategy == 'genetic':
            send_status({"status": "Starting DEAP genetic optimization...", "progress": 20})
            
            best_model, history = optimize_with_genetic_deap(
                X_train, y_train,
                actual_input_size, actual_output_size,
                model_info, population_size=20, generations=max_epochs, status_callback=send_status, device=device
            )
            send_status({"status": "Final training of best genetic model...", "progress": 75})
            train_loader_genetic = DataLoader(
                TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32, device=device),
                    torch.tensor(y_train, dtype=torch.float32, device=device)
                ), 
                batch_size=32, 
                shuffle=True
            )
            
            optimizer = optim.Adam(best_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            final_epochs = max_epochs*2
            for epoch in range(final_epochs):
                best_model.train()
                for batch_X, batch_y in train_loader_genetic:
                    optimizer.zero_grad()
                    outputs = best_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                
                if epoch % max(1, final_epochs // 5) == 0:  
                    progress = 75 + int((epoch / final_epochs) * 10)  
                    send_status({"status": f"Genetic backpropagation training: epoch {epoch+1}/{final_epochs}", "progress": progress})
            
            send_status({"status": "Evaluating genetic model performance...", "progress": 85})
            best_model = best_model.to(device)  
            best_model.eval()
            with torch.no_grad():
                all_outputs = []
                all_targets = []
                
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    outputs = best_model(batch_X)
                    all_outputs.append(outputs.cpu().numpy())
                    all_targets.append(batch_y.cpu().numpy())
                
                all_outputs = np.vstack(all_outputs)
                all_targets = np.vstack(all_targets)
                all_outputs_unscaled = scaler_y.inverse_transform(all_outputs)
                all_targets_unscaled = scaler_y.inverse_transform(all_targets)
                
                best_loss = mean_squared_error(all_targets_unscaled, all_outputs_unscaled)
                r2 = r2_score(all_targets_unscaled, all_outputs_unscaled)
            
            
            genetic_arch = getattr(best_model, '_genetic_architecture', {})
            layers = genetic_arch.get('layers', model_info.get('num_layers', 1))
            neurons = genetic_arch.get('neurons', model_info.get('max_width', 64))
            
            best_result = {
                "strategy": "Genetic", 
                "test_loss": float(best_loss), 
                "r2_score": float(r2), 
                "layers": int(layers), 
                "neurons": int(neurons),
                "enhancement_factor": genetic_arch.get('enhancement_factor', 1.0),
                "enhancement_pattern": genetic_arch.get('enhancement_pattern', 0),
                "history": history
            }
            results = [{
                "strategy": "Genetic", 
                "generations": max_epochs, 
                "fitness_history": history, 
                "test_loss": float(best_loss), 
                "r2_score": float(r2),
                "layers": int(layers),
                "neurons": int(neurons)
            }]
            
        else:
            send_status({"status": f"Unknown strategy: {strategy}", "error": True})
            return {"error": f"Unknown strategy: {strategy}"}
        best_model = best_model.to("cpu")
        dummy_input = torch.randn(1, actual_input_size, device="cpu")  
        onnx_path = os.path.join(tempfile.gettempdir(), "optimized_model.onnx")
        best_model.eval()
        try:
            logging.info(f"Exporting model to ONNX: {onnx_path}")
            torch.onnx.export(
                best_model, 
                dummy_input, 
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=12,
                do_constant_folding=True,
                export_params=True,
                keep_initializers_as_inputs=False,
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX
            )
            if not os.path.exists(onnx_path):
                raise RuntimeError(f"ONNX export failed: file not created at {onnx_path}")
            
            file_size = os.path.getsize(onnx_path)
            if file_size == 0:
                raise RuntimeError(f"ONNX export failed: empty file created at {onnx_path}")
            
            logging.info(f"✓ ONNX export successful: {file_size} bytes")
            
        except Exception as export_error:
            logging.exception(f"ONNX export failed: {export_error}")
            send_status({"status": f"ONNX export failed: {export_error}", "error": True})
            return {"error": f"ONNX export failed: {export_error}"}
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logging.info("✓ ONNX model validation passed")
        except Exception as load_error:
            logging.exception(f"ONNX model validation failed: {load_error}")
            send_status({"status": f"ONNX model validation failed: {load_error}", "error": True})
            return {"error": f"ONNX model validation failed: {load_error}"}
        ops_before = len(onnx_model.graph.node)
        node_types_before = {}
        for node in onnx_model.graph.node:
            node_types_before[node.op_type] = node_types_before.get(node.op_type, 0) + 1
        
        logging.info(f"ONNX model before optimization: {ops_before} operations")
        logging.info(f"Node types before: {dict(sorted(node_types_before.items(), key=lambda x: x[1], reverse=True)[:5])}")
        if HAS_ONNX_OPTIMIZER:
            try:
                optimization_passes = [
                    'eliminate_identity',
                    'eliminate_nop_transpose', 
                    'eliminate_nop_pad',
                    'eliminate_duplicate_initializer',
                    'eliminate_deadend',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_unsqueezes',
                    'eliminate_nop_reshape',
                    'eliminate_nop_dropout',
                    'fuse_add_bias_into_conv',
                    'eliminate_unused_initializer',
                    'fuse_matmul_add_bias_into_gemm',
                ]
                onnx_model = onnxoptimizer.optimize(onnx_model, optimization_passes)
                ops_after = len(onnx_model.graph.node)
                node_types_after = {}
                for node in onnx_model.graph.node:
                    node_types_after[node.op_type] = node_types_after.get(node.op_type, 0) + 1
                
                logging.info(f"✓ ONNX Optimizer applied successfully:")
                logging.info(f"  Operations: {ops_before} → {ops_after} (reduced by {ops_before - ops_after})")
                logging.info(f"  Node types after: {dict(sorted(node_types_after.items(), key=lambda x: x[1], reverse=True)[:5])}")
                logging.info(f"  Applied {len(optimization_passes)} optimization passes")
                excessive_ops = ['Slice', 'Unsqueeze', 'Concat', 'Squeeze', 'Reshape', 'Transpose']
                excessive_count_before = sum(node_types_before.get(op, 0) for op in excessive_ops)
                excessive_count_after = sum(node_types_after.get(op, 0) for op in excessive_ops)
                
                if excessive_count_after < excessive_count_before:
                    logging.info(f"  Excessive operations reduced: {excessive_count_before} → {excessive_count_after}")
                elif excessive_count_after > 10:
                    logging.warning(f"  Still {excessive_count_after} potentially excessive operations remaining")
                    
            except Exception as e:
                logging.warning(f"ONNX optimization failed: {e}")
                logging.info("Continuing with unoptimized model")
        else:
            logging.info("ONNX optimizer not available, skipping post-processing optimization")
        try:
            nodes_renamed = 0
            for i, node in enumerate(onnx_model.graph.node):
                if any(prefix in node.name for prefix in ['Universal', 'universal', '_layers_', '_adapter', 'enhanced_']):
                    base_name = node.op_type
                    existing_names = [n.name for n in onnx_model.graph.node if n != node]
                    if base_name in existing_names:
                        counter = 1
                        while f"{base_name}_{counter}" in existing_names:
                            counter += 1
                        node.name = f"{base_name}_{counter}"
                    else:
                        node.name = base_name
                    nodes_renamed += 1
            
            if nodes_renamed > 0:
                logging.info(f"✓ Post-export cleanup: Simplified {nodes_renamed} node names to operation types for cleaner visualization")
        except Exception as e:
            logging.warning(f"Node name cleanup failed: {e}")
        try:
            onnx.checker.check_model(onnx_model)
            onnx.save(onnx_model, onnx_path)
            if not os.path.exists(onnx_path):
                raise RuntimeError(f"Final ONNX save failed: file not found at {onnx_path}")
            
            final_file_size = os.path.getsize(onnx_path)
            if final_file_size == 0:
                raise RuntimeError(f"Final ONNX save failed: empty file at {onnx_path}")
            
            logging.info(f"✓ Final ONNX model validation and save passed: {final_file_size} bytes")
            
        except Exception as e:
            logging.exception(f"ONNX model final validation/save failed: {e}")
            send_status({"status": f"ONNX model save failed: {e}", "error": True})
            return {"error": f"ONNX model save failed: {e}"}
        final_ops = len(onnx_model.graph.node)
        final_node_types = {}
        for node in onnx_model.graph.node:
            final_node_types[node.op_type] = final_node_types.get(node.op_type, 0) + 1
        
        logging.info("=== Final ONNX Model Summary ===")
        logging.info(f"Total operations: {final_ops}")
        logging.info(f"Main operation types: {dict(sorted(final_node_types.items(), key=lambda x: x[1], reverse=True)[:10])}")
        logging.info("ONNX export and optimization complete")
        
        send_status({"status": "Optimization complete", "progress": 100})
        final_summary = {
            "best_loss": float(best_loss),
            "best_r2": float(best_result.get("r2_score", 0)),
            "models_tested": len(results) if strategy == 'brute-force' else f"{strategy}_optimization",
            "strategy_used": strategy
        }
        
        final_result = {
            "results": results,
            "best_config": best_result,
            "model_path": onnx_path,
            "summary": final_summary,
        }
        final_result = make_json_serializable(final_result)
        return final_result
        
    except Exception as e:
        logging.exception("Error in find_optimal_architecture")
        send_status({"status": f"Error: {str(e)}", "error": True, "progress": 0})
        return {"error": str(e)}

def download_optimized_model(path):
    try:
        if not os.path.exists(path):
            logging.error(f"Model file not found at path: {path}")
            return {"error": f"Model file not found at path: {path}"}
        file_size = os.path.getsize(path)
        if file_size == 0:
            logging.error(f"Model file is empty: {path}")
            return {"error": "Model file is empty"}
        
        logging.info(f"Reading model file: {path} (size: {file_size} bytes)")
        
        with open(path, "rb") as f:
            model_bytes = f.read()
        if not model_bytes:
            logging.error("No data read from model file")
            return {"error": "No data read from model file"}
        if len(model_bytes) != file_size:
            logging.error(f"Data length mismatch: expected {file_size}, got {len(model_bytes)}")
            return {"error": "Data length mismatch"}
        try:
            base64_data = base64.b64encode(model_bytes).decode("utf-8")
        except Exception as encode_error:
            logging.exception(f"Failed to encode model to base64: {encode_error}")
            return {"error": f"Failed to encode model to base64: {encode_error}"}
        if not base64_data:
            logging.error("Base64 encoding resulted in empty string")
            return {"error": "Base64 encoding failed"}
        
        
        try:
            test_decode = base64.b64decode(base64_data)
            if len(test_decode) != len(model_bytes):
                logging.error("Base64 round-trip validation failed")
                return {"error": "Base64 encoding validation failed"}
        except Exception as decode_error:
            logging.exception(f"Base64 validation failed: {decode_error}")
            return {"error": f"Base64 validation failed: {decode_error}"}
        
        logging.info(f"Successfully prepared model for download: {len(base64_data)} chars base64")
        
        return {
            "base64": base64_data,
            "filename": "optimized_model.onnx",
            "size": file_size,
            "base64_length": len(base64_data)
        }
    except Exception as e:
        logging.exception(f"Failed to read optimized model from {path}")
        return {"error": str(e)}

import os
import io
import base64
import logging
import json
import pandas as pd
import onnx
import torch
import tempfile
from typing import Optional

from Other.Data import analyze_csv_data
from Other.FileHandler import createTempFile
from Other.Utils import get_device, make_json_serializable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_onnx_model(file_bytes):
    """ONNX visualization only - kept minimal."""
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

        summary = {
            "ir_version": model.ir_version,
            "producer": model.producer_name,
            "inputs": [{"name": inp.name} for inp in graph.input],
            "outputs": [{"name": out.name} for out in graph.output],
            "node_count": len(graph.node)
        }

        os.remove(temp_file_path)
        return {"nodes": nodes, "edges": edges, "summary": summary}

    except Exception as e:
        logging.exception("Failed to load ONNX model for visualization")
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
        return make_json_serializable(result)
    except Exception as e:
        logging.exception("Failed to load CSV data")
        return {"error": str(e)}


def _dummy_shape_from_descriptor(input_shape):
    dummy_shape = []
    for index, dim in enumerate(input_shape):
        if dim is None or dim <= 0:
            dummy_shape.append(1 if index == 0 else 1)
        else:
            dummy_shape.append(int(dim))
    if not dummy_shape:
        dummy_shape = [1, 1]
    return dummy_shape

def export_descriptor_model_to_onnx(model, descriptor, output_dir, device=None):
    """
    Export the optimized reconstructed PyTorch module to a runnable ONNX file.
    """
    if device is None:
        device = torch.device("cpu")

    onnx_path = os.path.join(output_dir, "optimized_model.onnx")
    model_cpu = model.to("cpu")
    model_cpu.eval()

    dummy_shape = _dummy_shape_from_descriptor(descriptor.input_shape)
    dummy_input = torch.randn(*dummy_shape, dtype=torch.float32)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported)
    return onnx_path



def descriptor_to_upload_response(descriptor):
    """Generate graph visualization from descriptor."""
    nodes = [
        {"id": "input", "label": "input", "title": json.dumps({"shape": descriptor.input_shape}, indent=2), "group": "input"},
        {"id": "output", "label": "output", "title": json.dumps({"shape": descriptor.output_shape}, indent=2), "group": "output"},
    ]

    for node in descriptor.nodes:
        nodes.append({
            "id": node.id,
            "label": node.id,
            "title": json.dumps({
                "type": node.type,
                "params": node.params,
                "execution": getattr(node, 'execution', {}),
            }, indent=2),
            "group": node.type,
        })

    edges = [
        {"from": edge.source, "to": edge.target, "label": f"{getattr(edge, 'source_port', '')}→{getattr(edge, 'target_port', '')}"}
        for edge in descriptor.edges
    ]

    summary = {
        "format": "pt2_descriptor_bundle",
        "model_name": descriptor.model_name,
        "input_shape": descriptor.input_shape,
        "output_shape": descriptor.output_shape,
        "node_count": len(descriptor.nodes),
        "edge_count": len(descriptor.edges),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": make_json_serializable(summary),
    }

def process_uploaded_model(file_bytes, filename):
    """Modern entry point: prioritize descriptor bundles, ONNX only for viz."""
    from Core.ArchitectureDescriptor import ArchitectureDescriptor

    if filename.endswith((".pt", ".pth", ".pt2")):
        try:
            temp_in = createTempFile(file_bytes, ".pt")
            loaded = torch.load(temp_in, map_location="cpu", weights_only=False)
            os.remove(temp_in)

            if isinstance(loaded, dict):
                for key in ("model_config", "descriptor", "architecture"):
                    if key in loaded and isinstance(loaded[key], dict):
                        descriptor = ArchitectureDescriptor.from_dict(loaded[key])
                        descriptor.validate()
                        return descriptor_to_upload_response(descriptor)

                # Pure weights file
                if "state_dict" in loaded or all(isinstance(v, torch.Tensor) for v in loaded.values()):
                    return {
                        "summary": {
                            "format": "pytorch_weights",
                            "message": "Weights loaded successfully. Please upload model_config.json separately for full analysis.",
                            "has_state_dict": True,
                        },
                        "nodes": [],
                        "edges": [],
                    }

            return {"error": "Unsupported PyTorch file. Upload a bundle with model_config or use ONNX for visualization."}

        except Exception as e:
            logging.exception("Failed to process PyTorch bundle")
            return {"error": str(e)}

    else:
        # ONNX visualization only
        return load_onnx_model(file_bytes)

def _extract_state_dict_from_bytes(model_bytes, is_pytorch: bool):
    if not is_pytorch:
        return None

    temp_in = createTempFile(model_bytes, ".pt")
    try:
        checkpoint = torch.load(temp_in, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return checkpoint
        if hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
    except Exception as exc:
        logging.warning(f"Could not extract state dict from uploaded model: {exc}")
    finally:
        if os.path.exists(temp_in):
            os.remove(temp_in)
    return None

def _load_descriptor_from_bytes(model_bytes, is_pytorch: bool, input_dim: int, output_dim: int):
    from Core.ArchitectureDescriptor import ArchitectureDescriptor

    try:
        payload = json.loads(model_bytes.decode("utf-8"))
        if isinstance(payload, dict) and "nodes" in payload:
            descriptor = ArchitectureDescriptor.from_dict(payload)
            descriptor.validate()
            return descriptor
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError):
        pass

    if is_pytorch:
        temp_in = createTempFile(model_bytes, ".pt")
        try:
            checkpoint = torch.load(temp_in, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                for key in ("model_config", "descriptor", "architecture"):
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        descriptor = ArchitectureDescriptor.from_dict(checkpoint[key])
                        descriptor.validate()
                        return descriptor
        except Exception as exc:
            logging.warning(f"Could not load descriptor from checkpoint: {exc}")
        finally:
            if os.path.exists(temp_in):
                os.remove(temp_in)

    descriptor = ArchitectureDescriptor.default_feedforward(input_dim, output_dim)
    descriptor.validate()
    return descriptor

def find_optimal_architecture(onnx_bytes, input_features, target_feature, df: pd.DataFrame, max_epochs=5, 
                            strategy='brute-force', generations=5, status_callback=None, is_pytorch=False,
                            weights_bytes=None):
    """
    Descriptor-first neuroevolution pipeline.
    Uses model_config.json (or embedded descriptor) plus optional weights as inputs.
    """
    from Core.ArchitectureDescriptor import ArchitectureDescriptor
    from Optimize.Neuroevolution import NeuroevolutionEngine
    from Data.DataPipeline import DataPipeline
    from Core.AdaptedModel import AdaptedModel
    
    def send_status(message):
        if status_callback:
            status_callback(message)
        else:
            logging.info(message.get("status", str(message)))

    temp_csv = None
    try:
        device = get_device()
        torch.set_num_threads(4)
        
        if df is None:
            send_status({"status": "No data loaded", "error": True})
            return {"error": "No data loaded"}

        if onnx_bytes is None:
            send_status({"status": "No model loaded", "error": True})
            return {"error": "No model loaded"}

        feature_cols = input_features if isinstance(input_features, list) else [input_features]
        target_cols = target_feature if isinstance(target_feature, list) else [target_feature]
        input_dim = len(feature_cols)
        output_dim = len(target_cols)

        send_status({"status": "Loading architecture descriptor...", "progress": 5})
        initial_descriptor = _load_descriptor_from_bytes(onnx_bytes, is_pytorch, input_dim, output_dim)
        parent_state_dict = None
        if weights_bytes is not None:
            send_status({"status": "Loading parent weights...", "progress": 7})
            parent_state_dict = _extract_state_dict_from_bytes(weights_bytes, True)
        elif is_pytorch:
            parent_state_dict = _extract_state_dict_from_bytes(onnx_bytes, is_pytorch)

        temp_csv = os.path.join(tempfile.gettempdir(), "temp_dataset.csv")
        df.to_csv(temp_csv, index=False)

        # Detect sequence_length if model expects 3D inputs [batch, sequence_length, features]
        sequence_length = None
        if len(initial_descriptor.input_shape) == 3:
            sequence_length = initial_descriptor.input_shape[1]

        send_status({"status": "Preparing leakage-free data pipeline...", "progress": 10})
        pipeline = DataPipeline.prepare_data(
            temp_csv,
            feature_cols,
            target_cols,
            problem_type="regression",
            batch_size=32 if device.type == "cpu" else min(128, max(32, len(df) // 8)),
            sequence_length=sequence_length,
        )

        if initial_descriptor.input_shape[-1] != pipeline.input_shape[-1]:
            initial_descriptor.input_shape = pipeline.input_shape
            for node in initial_descriptor.nodes:
                if node.type == "Linear" and any(
                    edge.source == "input" and edge.target == node.id for edge in initial_descriptor.edges
                ):
                    node.params["in_features"] = pipeline.input_shape[-1]

        if initial_descriptor.output_shape[-1] != pipeline.output_shape[-1]:
            initial_descriptor.output_shape = pipeline.output_shape
            output_nodes = [edge.source for edge in initial_descriptor.edges if edge.target == "output"]
            if output_nodes:
                out_node = next(n for n in initial_descriptor.nodes if n.id == output_nodes[0])
                if out_node.type == "Linear":
                    out_node.params["out_features"] = pipeline.output_shape[-1]

        initial_descriptor.validate()

        send_status({"status": "Starting neuroevolution...", "progress": 20})
        population_size = min(20, max(4, max_epochs * 4))
        engine = NeuroevolutionEngine(initial_descriptor, population_size=population_size)
        best_descriptor, best_model = engine.evolve(
            train_loader=pipeline.train_loader,
            val_loader=pipeline.val_loader,
            generations=max(2, generations),
            max_epochs=max_epochs,
            device=str(device),
            parent_state_dict=parent_state_dict,
        )

        expected_input = best_descriptor.input_shape[-1]
        actual_input = pipeline.input_shape[-1]
        expected_output = best_descriptor.output_shape[-1]
        actual_output = pipeline.output_shape[-1]
        if expected_input != actual_input or expected_output != actual_output:
            best_model = AdaptedModel.from_shape_mismatch(
                best_model,
                actual_input_dim=actual_input,
                expected_input_dim=expected_input,
                actual_output_dim=actual_output,
                expected_output_dim=expected_output,
            )

        send_status({"status": "Saving optimized model bundle...", "progress": 90})
        output_dir = tempfile.mkdtemp(prefix="optimized_model_")
        config_path = os.path.join(output_dir, "model_config.json")
        weights_path = os.path.join(output_dir, "model_weights.pth")
        onnx_path = os.path.join(output_dir, "optimized_model.onnx")

        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(best_descriptor.to_json())

        torch.save(
            {
                "state_dict": best_model.state_dict(),
                "model_config": best_descriptor.to_dict(),
            },
            weights_path,
        )

        bundle_path = os.path.join(output_dir, "optimized_model.pt2")
        torch.save(
            {
                "state_dict": best_model.state_dict(),
                "model_config": best_descriptor.to_dict(),
            },
            bundle_path,
        )

        try:
            onnx_path = export_descriptor_model_to_onnx(best_model, best_descriptor, output_dir, device=device)
        except Exception as export_error:
            logging.exception("Failed to export optimized model to ONNX")
            onnx_path = None

        send_status({"status": "Optimization complete", "progress": 100})
        return {
            "status": "success",
            "model_path": bundle_path,
            "model_config_path": config_path,
            "model_weights_path": weights_path,
            "model_onnx_path": onnx_path,
            "onnx_exported": onnx_path is not None,
            "summary": {
                "strategy_used": "neuroevolution",
                "requested_strategy": strategy,
                "generations": max(2, generations),
                "population_size": population_size,
                "input_features": feature_cols,
                "output_features": target_cols,
            },
            "best_config": best_descriptor.to_dict(),
        }
        
    except Exception as e:
        logging.exception("Error in find_optimal_architecture")
        send_status({"status": f"Error: {str(e)}", "error": True, "progress": 0})
        return {"error": str(e)}
    finally:
        if temp_csv and os.path.exists(temp_csv):
            os.remove(temp_csv)

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
        
        filename = os.path.basename(path)
        if not filename:
            filename = "optimized_model.pt2"

        return {
            "base64": base64_data,
            "filename": filename,
            "size": file_size,
            "base64_length": len(base64_data)
        }
    except Exception as e:
        logging.exception(f"Failed to read optimized model from {path}")
        return {"error": str(e)}


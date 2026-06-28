# python -m uvicorn App:app --reload --host 127.0.0.1 --port 8000

from fastapi import FastAPI, Request, Header, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
import json
from pydantic import BaseModel
import os
import concurrent.futures
import io
from dotenv import load_dotenv
import hashlib
from typing import Optional

from TwoN2D import (
    load_onnx_model,
    process_uploaded_model,
    load_csv_data,
    find_optimal_architecture,
    download_optimized_model
)

from Other.FileHandler import (
    getFileBinaryData,
    uploadFile
)

from Other.Data import (encode_data_for_ml, map_original_to_encoded_columns)

load_dotenv()
API_KEY_HASH = os.getenv("API_KEY_HASH")

class FilePathRequest(BaseModel):
    filepath: str

app = FastAPI(
    title="2N2D API",
    description="API for uploading models, CSVs, optimizing ONNX models, and tracking optimization status.",
    version="1.0.0",
    contact={
        "name": "2N2D Team",
        "email": "support@2n2d.com"
    },
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow requests from your frontend (adjust if deploying)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

message_queues = {}

def verify_api_key(x_api_key: str = Header(..., description="Your API key")):
    """
    Verifies the provided API key against the stored hash.
    """
    if not API_KEY_HASH:
        raise HTTPException(status_code=500, detail="API key hash not configured.")
    if x_api_key != API_KEY_HASH:
        raise HTTPException(status_code=401, detail="Invalid API key.")

def parse_feature_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, str):
                return [parsed]
        except json.JSONDecodeError:
            pass
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return [value]

def parse_target_feature(value):
    parsed = parse_feature_list(value)
    return parsed if len(parsed) > 1 else (parsed[0] if parsed else value)

def encode_dataframe_for_request(df: pd.DataFrame, input_features, target_feature, encoding):
    if encoding not in ("label", "onehot"):
        encoded_df = df
        encoding_metadata = {}
    else:
        result = encode_data_for_ml(df, encoding_type=encoding)
        encoded_df = result[0]
        encoding_metadata = result[1]

    mapped_input_features = map_original_to_encoded_columns(input_features, encoding_metadata, encoded_df)
    target_features = target_feature if isinstance(target_feature, list) else [target_feature]
    mapped_target_features = map_original_to_encoded_columns(target_features, encoding_metadata, encoded_df)
    mapped_target_feature = mapped_target_features if isinstance(target_feature, list) else mapped_target_features[0]
    return encoded_df, mapped_input_features, mapped_target_feature

async def attach_optimized_file_urls(result: dict, session_id: str):
    if result.get("model_path"):
        filename = os.path.basename(result["model_path"])
        uploaded_url = await uploadFile(result["model_path"], f"{session_id}/optim")
        result["url"] = f"{session_id}/optim/{filename}"
        result["public_url"] = uploaded_url

    if result.get("model_onnx_path"):
        onnx_filename = os.path.basename(result["model_onnx_path"])
        onnx_uploaded_url = await uploadFile(result["model_onnx_path"], f"{session_id}/optim")
        result["onnx_url"] = f"{session_id}/optim/{onnx_filename}"
        result["onnx_public_url"] = onnx_uploaded_url

    return result

@app.get("/", tags=["General"], summary="Root endpoint", description="Check if the API is running.")
def root():
    
    return {"message": "2N2D API is running"}

@app.post("/upload-model", tags=["Data Handling"], summary="Upload ONNX or PyTorch Model", description="Upload an ONNX or PyTorch model file for processing.", dependencies=[Depends(verify_api_key)])
async def upload_model(request: FilePathRequest, session_id: Optional[str] = Header(None, description="Session identifier")):
    """
    Upload an ONNX or PyTorch model file for processing.
    """
    binary_data = await getFileBinaryData(request.filepath)
    filename = os.path.basename(request.filepath)
    result = process_uploaded_model(binary_data, filename)
    return JSONResponse(content=result)

@app.post("/upload-csv", tags=["Data Handling"], summary="Upload CSV Data", description="Upload a CSV file for processing.", dependencies=[Depends(verify_api_key)])
async def upload_csv(request: FilePathRequest, session_id: Optional[str] = Header(None, description="Session identifier")):
    """
    Upload a CSV file for processing.
    """
    binary_data = await getFileBinaryData(request.filepath)
    result = load_csv_data(binary_data, os.path.basename(request.filepath))
    return JSONResponse(content=result)

@app.post("/optimize", tags=["Optimization"], summary="Optimize Model", description="Optimize a model bundle or descriptor using provided CSV data and parameters.", dependencies=[Depends(verify_api_key)])
async def optimize(
    request: dict,
    session_id: str = Header(..., description="Session identifier")
):
    """
    Optimize a model bundle/descriptor using provided CSV data and parameters.
    Accepts model_path, bundle_path, pt2_path, or the legacy onnx_path field.
    """
    queue = message_queues.get(session_id)
    if queue is None:
        queue = []
        message_queues[session_id] = queue

    message_queues[session_id].append("Processing request...")
    input_features = request.get("input_features")
    target_feature = request.get("target_feature")
    epochs = request.get("max_epochs", 5)
    generations = request.get("generations", 5)
    csv_path = request.get("csv_path")
    model_path = (
        request.get("model_path")
        or request.get("bundle_path")
        or request.get("pt2_path")
        or request.get("onnx_path")
    )
    encoding = request.get("encoding")
    strat = request.get("strategy")

    if not csv_path:
        raise HTTPException(status_code=400, detail="csv_path is required.")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required. You may also use bundle_path, pt2_path, or legacy onnx_path.")

    message_queues[session_id].append({
        "status": "Downloading csv data from database...",
        "progress": 3
    })
    csv_binary = await getFileBinaryData(csv_path)
    df = pd.read_csv(io.BytesIO(csv_binary))
    input_features = parse_feature_list(input_features)
    target_feature = parse_target_feature(target_feature)
    encodedDf, mapped_input_features, mapped_target_feature = encode_dataframe_for_request(
        df, input_features, target_feature, encoding
    )

    message_queues[session_id].append({
        "status": "Encoding csv data...",
        "progress": 5
    })

    message_queues[session_id].append({
        "status": "Downloading model data from database...",
        "progress": 10
    })
    model_binary = await getFileBinaryData(model_path)
    
    is_pytorch_model = model_path.endswith(".pt") or model_path.endswith(".pth") or model_path.endswith(".pt2")

    def status_callback(message):
        message_queues[session_id].append(message)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: find_optimal_architecture(
                onnx_bytes=model_binary,
                df=encodedDf,
                input_features=mapped_input_features,
                target_feature=mapped_target_feature,
                status_callback=status_callback,
                max_epochs=epochs,
                strategy=strat,
                generations=generations,
                is_pytorch=is_pytorch_model
            )
        )

    if "model_path" not in result:
        return JSONResponse(content=result, status_code=400)

    result = await attach_optimized_file_urls(result, session_id)

    return JSONResponse(content=result)

@app.post(
    "/optimize-bundle-upload",
    tags=["Optimization"],
    summary="Optimize from uploaded .pt2 bundle and CSV",
    description="Upload one .pt2/.pt/.pth bundle containing model_config and state_dict, plus CSV data, then run neuroevolution.",
    dependencies=[Depends(verify_api_key)]
)
async def optimize_bundle_upload(
    bundle_file: UploadFile = File(..., description="PyTorch bundle containing model_config and state_dict, usually .pt2"),
    csv_file: UploadFile = File(..., description="CSV dataset used for optimization"),
    input_features: str = Form(..., description='JSON array or comma-separated list, e.g. ["f1","f2"]'),
    target_feature: str = Form(..., description='Target column name or JSON array for multiple targets'),
    max_epochs: int = Form(5),
    generations: int = Form(5),
    encoding: Optional[str] = Form(None),
    strategy: str = Form("neuroevolution"),
    session_id: str = Header(..., description="Session identifier")
):
    queue = message_queues.get(session_id)
    if queue is None:
        queue = []
        message_queues[session_id] = queue

    message_queues[session_id].append("Processing uploaded bundle optimization request...")
    feature_cols = parse_feature_list(input_features)
    target_cols = parse_target_feature(target_feature)

    if not feature_cols:
        raise HTTPException(status_code=400, detail="input_features must contain at least one feature column.")
    if not target_cols:
        raise HTTPException(status_code=400, detail="target_feature is required.")

    bundle_bytes = await bundle_file.read()
    csv_bytes = await csv_file.read()

    if not bundle_bytes:
        raise HTTPException(status_code=400, detail="bundle_file is empty.")
    if not csv_bytes:
        raise HTTPException(status_code=400, detail="csv_file is empty.")

    message_queues[session_id].append({
        "status": "Encoding csv data...",
        "progress": 5
    })
    df = pd.read_csv(io.BytesIO(csv_bytes))
    encoded_df, mapped_input_features, mapped_target_feature = encode_dataframe_for_request(
        df, feature_cols, target_cols, encoding
    )

    def status_callback(message):
        message_queues[session_id].append(message)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: find_optimal_architecture(
                onnx_bytes=bundle_bytes,
                df=encoded_df,
                input_features=mapped_input_features,
                target_feature=mapped_target_feature,
                status_callback=status_callback,
                max_epochs=max_epochs,
                strategy=strategy,
                generations=generations,
                is_pytorch=True
            )
        )

    if "model_path" not in result:
        return JSONResponse(content=result, status_code=400)

    result = await attach_optimized_file_urls(result, session_id)

    return JSONResponse(content=result)

@app.post(
    "/optimize-upload",
    tags=["Optimization"],
    summary="Optimize from uploaded descriptor, weights, and CSV",
    description="Upload model_config.json and model_weights.pth as separate files, plus CSV data, then run neuroevolution.",
    dependencies=[Depends(verify_api_key)]
)
async def optimize_upload(
    config_file: UploadFile = File(..., description="Architecture descriptor JSON, usually model_config.json"),
    weights_file: UploadFile = File(..., description="PyTorch weights/checkpoint file, usually model_weights.pth"),
    csv_file: UploadFile = File(..., description="CSV dataset used for optimization"),
    input_features: str = Form(..., description='JSON array or comma-separated list, e.g. ["f1","f2"]'),
    target_feature: str = Form(..., description='Target column name or JSON array for multiple targets'),
    max_epochs: int = Form(5),
    generations: int = Form(5),
    encoding: Optional[str] = Form(None),
    strategy: str = Form("neuroevolution"),
    session_id: str = Header(..., description="Session identifier")
):
    queue = message_queues.get(session_id)
    if queue is None:
        queue = []
        message_queues[session_id] = queue

    message_queues[session_id].append("Processing uploaded optimization request...")
    feature_cols = parse_feature_list(input_features)
    target_cols = parse_target_feature(target_feature)

    if not feature_cols:
        raise HTTPException(status_code=400, detail="input_features must contain at least one feature column.")
    if not target_cols:
        raise HTTPException(status_code=400, detail="target_feature is required.")

    config_bytes = await config_file.read()
    weights_bytes = await weights_file.read()
    csv_bytes = await csv_file.read()

    if not config_bytes:
        raise HTTPException(status_code=400, detail="config_file is empty.")
    if not weights_bytes:
        raise HTTPException(status_code=400, detail="weights_file is empty.")
    if not csv_bytes:
        raise HTTPException(status_code=400, detail="csv_file is empty.")

    message_queues[session_id].append({
        "status": "Encoding csv data...",
        "progress": 5
    })
    df = pd.read_csv(io.BytesIO(csv_bytes))
    encoded_df, mapped_input_features, mapped_target_feature = encode_dataframe_for_request(
        df, feature_cols, target_cols, encoding
    )

    def status_callback(message):
        message_queues[session_id].append(message)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: find_optimal_architecture(
                onnx_bytes=config_bytes,
                weights_bytes=weights_bytes,
                df=encoded_df,
                input_features=mapped_input_features,
                target_feature=mapped_target_feature,
                status_callback=status_callback,
                max_epochs=max_epochs,
                strategy=strategy,
                generations=generations,
                is_pytorch=False
            )
        )

    if "model_path" not in result:
        return JSONResponse(content=result, status_code=400)

    result = await attach_optimized_file_urls(result, session_id)

    return JSONResponse(content=result)

@app.get("/optimization-status/{session_id}", dependencies=[Depends(verify_api_key)])
async def stream_status(session_id: str, request: Request):
    queue = message_queues.get(session_id)
    if queue is None:
        queue = []
        message_queues[session_id] = queue

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            if queue:
                message = queue.pop(0)
                yield f"data: {json.dumps(message)}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/download-optimized", tags=["Optimization"], summary="Download Optimized Model", description="Download the optimized ONNX model.", dependencies=[Depends(verify_api_key)])
def download_optimized(file_path: str, session_id: str = Header(..., description="Session identifier")):
    """
    Download the optimized ONNX model.
    """
    result = download_optimized_model(file_path)
    return JSONResponse(content=result)

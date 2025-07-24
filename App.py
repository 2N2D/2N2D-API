from fastapi import FastAPI, Request, Header, HTTPException, Depends
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

from TwoN2D import (
    load_onnx_model,
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

@app.get("/", tags=["General"], summary="Root endpoint", description="Check if the API is running.")
def root():
    """
    Returns a simple message to confirm the API is running.
    """
    return {"message": "2N2D API is running"}

@app.post("/upload-model", tags=["Data Handling"], summary="Upload ONNX Model", description="Upload an ONNX model file for processing.", dependencies=[Depends(verify_api_key)])
async def upload_model(request: FilePathRequest, session_id: str = Header(..., description="Session identifier")):
    """
    Upload an ONNX model file for processing.
    """
    binary_data = await getFileBinaryData(request.filepath)
    result = load_onnx_model(binary_data)
    return JSONResponse(content=result)

@app.post("/upload-csv", tags=["Data Handling"], summary="Upload CSV Data", description="Upload a CSV file for processing.", dependencies=[Depends(verify_api_key)])
async def upload_csv(request: FilePathRequest, session_id: str = Header(..., description="Session identifier")):
    """
    Upload a CSV file for processing.
    """
    binary_data = await getFileBinaryData(request.filepath)
    result = load_csv_data(binary_data, os.path.basename(request.filepath))
    return JSONResponse(content=result)

@app.post("/optimize", tags=["Optimization"], summary="Optimize Model", description="Optimize an ONNX model using provided CSV data and parameters.", dependencies=[Depends(verify_api_key)])
async def optimize(
    request: dict,
    session_id: str = Header(..., description="Session identifier")
):
    """
    Optimize an ONNX model using provided CSV data and parameters.
    """
    queue = message_queues.get(session_id)
    if queue is None:
        queue = []
        message_queues[session_id] = queue

    message_queues[session_id].append("Processing request...")
    input_features = request.get("input_features")
    target_feature = request.get("target_feature")
    epochs = request.get("max_epochs")
    sessionId = request.get("session_id")
    csv_path = request.get("csv_path")
    onnx_path = request.get("onnx_path")
    encoding = request.get("encoding")
    strat = request.get("strategy")

    message_queues[session_id].append({
        "status": "Downloading csv data from database...",
        "progress": 3
    })
    csv_binary = await getFileBinaryData(csv_path)
    encodedDf = None
    encoding_metadata = None
    df = pd.read_csv(io.BytesIO(csv_binary))
    if encoding != "label" and encoding != "onehot":
        encodedDf = df
        encoding_metadata = {}
    else:
        result = encode_data_for_ml(df, encoding_type=encoding)
        encodedDf = result[0]
        encoding_metadata = result[1]
    
    mapped_input_features = map_original_to_encoded_columns(input_features, encoding_metadata, encodedDf)
    mapped_target_feature = map_original_to_encoded_columns([target_feature], encoding_metadata, encodedDf)[0]

    message_queues[session_id].append({
        "status": "Encoding csv data...",
        "progress": 5
    })

    message_queues[session_id].append({
        "status": "Downloading onnx data from database...",
        "progress": 10
    })
    onnx_binary = await getFileBinaryData(onnx_path)

    def status_callback(message):
        message_queues[session_id].append(message)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: find_optimal_architecture(
                onnx_bytes=onnx_binary,
                df=encodedDf,
                input_features=mapped_input_features,
                target_feature=mapped_target_feature,
                status_callback=status_callback,
                max_epochs=epochs,
                strategy=strat
            )
        )

    if "model_path" not in result:
        return JSONResponse(content=result, status_code=400)

    filename = os.path.basename(result["model_path"])
    await uploadFile(result["model_path"], f"{session_id}/optim")
    result["url"] = f"{session_id}/optim/{filename}"

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

@app.post("/headerTest", tags=["General"], summary="Header Test", description="Test endpoint for session_id header.", dependencies=[Depends(verify_api_key)])
def headerTest(session_id: str = Header(..., description="Session identifier")):
    """
    Test endpoint for session_id header.
    """
    return {"session_id": session_id}

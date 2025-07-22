# Run with:
# uvicorn --app-dir . TwoN2DEndPoint:app --reload --log-level debug

from fastapi import FastAPI, Request, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import pandas as pd
import json
from pydantic import BaseModel
import os
import concurrent.futures
import io

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


class FilePathRequest(BaseModel):
    filepath: str


app = FastAPI()

# Allow requests from your frontend (adjust if deploying)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

message_queues = {}


@app.get("/")
def root():
    return {"message": "2N2D API is running"}


# data handling

@app.post("/upload-model")
async def upload_model(request: FilePathRequest, session_id: str = Header(...)):
    binary_data = await getFileBinaryData(request.filepath, "onnx")
    result = load_onnx_model(binary_data)

    return JSONResponse(content=result)


@app.post("/upload-csv")
async def upload_csv(request: FilePathRequest, session_id: str = Header(...)):
    binary_data = await getFileBinaryData(request.filepath, "csv")
    result = load_csv_data(binary_data, os.path.basename(request.filepath))
    return JSONResponse(content=result)


# Optimization service

@app.post("/optimize")
async def optimize(request: dict, session_id: str = Header(...)):
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

    print(csv_path)

    message_queues[session_id].append({
        "status": "Downloading csv data from database...",
        "progress": 3
    })
    csv_binary = await getFileBinaryData(csv_path, "csv")
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
    onnx_binary = await getFileBinaryData(onnx_path, "onnx")

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
    await uploadFile(result["model_path"], f"{session_id}/{sessionId}/{filename}")
    result["url"] = f"{session_id}/{sessionId}/{filename}"

    return JSONResponse(content=result)


@app.get("/optimization-status/{session_id}")
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


@app.get("/download-optimized")
def download_optimized(file_path: str, session_id: str = Header(...)):
    result = download_optimized_model(file_path)
    return JSONResponse(content=result)


@app.post("/headerTest")
def headerTest(session_id: str = Header(...)):
    return {"session_id": session_id}

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pytest
from fastapi.testclient import TestClient
from App import app


client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "2N2D API is running"}

def test_upload_model_endpoint():
    headers = {
        "x-api-key": "51056f90d0dc5575a0bc848afddbbe8bffe0ee3e885d8c25793625b0c4688b61",
        "session-id": "yPLxSonrtJNGkAtv16Z6HxYT6kU2"
    }
    payload = {
        "filepath": "yPLxSonrtJNGkAtv16Z6HxYT6kU2/onnx/ensemble_policy.onnx"
    }
    response = client.post("/upload-model", json=payload, headers=headers)
    
    assert response.status_code == 200

    response_data = response.json()
    assert isinstance(response_data, dict)

def test_redoc_endpoint():
    response = client.get("/redoc")
    assert response.status_code == 200

def test_protected_endpoint_requires_auth():
    response = client.post("/headerTest")
    assert response.status_code == 422
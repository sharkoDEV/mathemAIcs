import io
import os
import re
import shutil
import subprocess
import sys
import threading
from typing import Dict, Optional

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from inference import Predictor

app = FastAPI(title="Synthetic Math AI")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOSS_PLOT = os.path.join(CHECKPOINT_DIR, "loss.png")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ACTIVE_MODEL = os.path.join(MODEL_DIR, "active.mathai")
os.makedirs(MODEL_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
_predictor: Optional[Predictor] = None
_train_state: Dict = {
    "running": False,
    "message": "idle",
    "returncode": None,
    "progress": 0.0,
    "logs": [],
}
_train_lock = threading.Lock()
_build_state: Dict = {"running": False, "message": "idle", "returncode": None}
_build_lock = threading.Lock()
_build_params = {"geometry": 200, "charts": 200, "ocr": 50, "lines": 40, "text": 400}
_train_params = {"epochs": 5}


class BuildDatasetRequest(BaseModel):
    geometry: int = Field(200, ge=1, description="Number of geometry images to generate")
    charts: int = Field(200, ge=1, description="Number of chart images to generate")
    ocr: int = Field(50, ge=1, description="Number of OCR text panel images to generate")
    lines: int = Field(40, ge=1, description="Number of line reasoning images to generate")
    text: int = Field(400, ge=10, description="Number of text-only prompts to create")


class TrainingRequest(BaseModel):
    epochs: int = Field(5, ge=1, description="Number of epochs to train for")


@app.on_event("startup")
async def load_model():
    global _predictor
    checkpoint_path = os.getenv("MODEL_FILE", ACTIVE_MODEL)
    if not os.path.exists(checkpoint_path):
        print(f"Model file {checkpoint_path} not found. API will load when available.")
        _predictor = None
        return
    _predictor = Predictor(checkpoint_path)


def refresh_predictor_if_available():
    global _predictor
    if os.path.exists(ACTIVE_MODEL):
        try:
            _predictor = Predictor(ACTIVE_MODEL)
        except Exception as exc:
            print(f"Failed to load model: {exc}")
            _predictor = None
    else:
        _predictor = None


@app.get("/")
async def root_page():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.post("/predict")
async def predict_endpoint(
    file: Optional[UploadFile] = File(None),
    prompt: str = Form("")
):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not ready.")
    image = None
    if file is not None:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    result = _predictor.predict(prompt=prompt, image=image)
    return result


def _train_worker():
    global _train_state
    with _train_lock:
        _train_state.update(
            {
                "running": True,
                "message": "training in progress",
                "returncode": None,
                "progress": 0.0,
                "logs": [],
            }
        )
        params = _train_params.copy()
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "src", "train.py")]
    cmd.extend(["--epochs", str(params.get("epochs", 5)), "--model_file", ACTIVE_MODEL])
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)")
    try:
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            with _train_lock:
                logs = _train_state.get("logs", [])
                logs.append(clean)
                _train_state["logs"] = logs[-200:]
                match = epoch_pattern.search(clean)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    _train_state["progress"] = min(current / max(total, 1), 1.0)
                _train_state["message"] = clean or _train_state["message"]
    finally:
        process.wait()
        success = process.returncode == 0
        message = "training complete" if success else f"training failed ({process.returncode})"
        with _train_lock:
            _train_state.update(
                {
                    "running": False,
                    "message": message,
                    "returncode": process.returncode,
                    "progress": 1.0 if success else _train_state.get("progress", 0.0),
                }
            )
        if success:
            refresh_predictor_if_available()


@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks, payload: TrainingRequest | None = Body(default=None)):
    if _train_state["running"]:
        return JSONResponse({"status": "running", "message": "training already in progress"}, status_code=200)
    if payload is not None:
        with _train_lock:
            _train_params.update(payload.dict())
            _train_state["message"] = f"queued training epochs={payload.epochs}"
    background_tasks.add_task(_train_worker)
    return {"status": "started", "message": "training launched"}


@app.get("/train/status")
async def training_status():
    return _train_state


@app.get("/train/logs")
async def training_logs():
    with _train_lock:
        return {"logs": _train_state.get("logs", []), "running": _train_state.get("running", False)}


@app.get("/train/plot")
async def training_plot():
    if not os.path.exists(LOSS_PLOT):
        raise HTTPException(status_code=404, detail="Loss plot not found")
    return FileResponse(LOSS_PLOT, media_type="image/png")


def _build_worker():
    global _build_state
    with _build_lock:
        _build_state.update({"running": True, "message": "building dataset", "returncode": None})
        params = _build_params.copy()
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "src", "build_dataset.py"),
        "--geometry",
        str(params.get("geometry", 200)),
        "--charts",
        str(params.get("charts", 200)),
        "--ocr",
        str(params.get("ocr", 50)),
        "--lines",
        str(params.get("lines", 40)),
        "--text",
        str(params.get("text", 400)),
    ]
    process = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    message = "dataset ready" if process.returncode == 0 else f"dataset build failed ({process.returncode})"
    if process.returncode != 0:
        message += f": {process.stderr[-400:]}"
    with _build_lock:
        _build_state.update({"running": False, "message": message, "returncode": process.returncode})


@app.post("/dataset/build")
async def build_dataset(request: BuildDatasetRequest, background_tasks: BackgroundTasks):
    if _build_state["running"]:
        return JSONResponse({"status": "running", "message": "dataset build already running"})
    with _build_lock:
        _build_params.update(request.dict())
        _build_state.update(
            {
                "message": (
                    "queued build geometry="
                    f"{request.geometry} charts={request.charts} ocr={request.ocr} lines={request.lines} text={request.text}"
                )
            }
        )
    background_tasks.add_task(_build_worker)
    return {"status": "started", "message": "dataset builder launched"}


@app.get("/dataset/status")
async def dataset_status():
    return _build_state


@app.get("/model/status")
async def model_status():
    exists = os.path.exists(ACTIVE_MODEL)
    size = os.path.getsize(ACTIVE_MODEL) if exists else 0
    return {
        "exists": exists,
        "filename": os.path.basename(ACTIVE_MODEL) if exists else None,
        "size": size,
    }


@app.get("/model/download")
async def download_model():
    if not os.path.exists(ACTIVE_MODEL):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(ACTIVE_MODEL, filename=os.path.basename(ACTIVE_MODEL), media_type="application/octet-stream")


@app.post("/model/upload")
async def upload_model(file: UploadFile = File(...)):
    contents = await file.read()
    with open(ACTIVE_MODEL, "wb") as f:
        f.write(contents)
    refresh_predictor_if_available()
    return {"status": "uploaded", "message": "Model uploaded successfully."}


@app.delete("/model")
async def delete_model():
    if os.path.exists(ACTIVE_MODEL):
        os.remove(ACTIVE_MODEL)
    refresh_predictor_if_available()
    return {"status": "deleted", "message": "Model file removed."}

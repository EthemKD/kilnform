"""Kilnform local AI backend — listens on 127.0.0.1 only, closed to the outside."""
import base64
import io
import sys
import threading

if sys.platform == "win32":
    # Windows EcoQoS throttles background/minimized console processes onto
    # efficiency cores — a make measured 4x slower. Opt this process out.
    try:
        import ctypes

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong),
            ]

        _PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        _state = _PowerThrottlingState(1, _PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)
        ctypes.windll.kernel32.SetProcessInformation(
            ctypes.windll.kernel32.GetCurrentProcess(),
            4,  # ProcessPowerThrottling
            ctypes.byref(_state),
            ctypes.sizeof(_state),
        )
    except Exception:
        pass

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()  # phone photos (.heic/.heif) become uploadable
except ImportError:
    pass

import pipeline


def _friendly(exc: Exception) -> HTTPException:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        torch.cuda.empty_cache()
        return HTTPException(507, "GPU memory is full — close other GPU-heavy apps and try again")
    return HTTPException(500, f"generation failed: {exc}")

app = FastAPI(title="Kilnform AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_warm = {"state": "cold"}  # cold | loading | ready | error
_lock = threading.Lock()


def _ensure_warm():
    with _lock:
        if _warm["state"] == "ready":
            return
        _warm["state"] = "loading"
        try:
            pipeline.warmup()
            _warm["state"] = "ready"
        except Exception as e:
            _warm["state"] = "error"
            _warm["error"] = str(e)
            raise


class TextRequest(BaseModel):
    prompt: str
    seed: int | None = None
    resolution: int = 256
    tier: str = "mc"  # "mc" = TripoSR at `resolution`; "ultra" = Hunyuan3D-2mini


def _pack(result):
    return {
        "glb_b64": base64.b64encode(result["glb"]).decode(),
        "preview_b64": base64.b64encode(pipeline.png_bytes(result["preview"])).decode(),
        "prompt_en": result["prompt_en"],
        "seconds": result["seconds"],
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "models": _warm["state"],
    }


@app.get("/api/progress")
def progress():
    return {"stage": pipeline.STAGE}


@app.post("/api/warmup")
def warmup():
    _ensure_warm()
    return {"models": _warm["state"]}


@app.post("/api/text-to-3d")
def text_to_3d(req: TextRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    if req.tier not in ("mc", "ultra"):
        raise HTTPException(400, "tier must be 'mc' or 'ultra'")
    _ensure_warm()
    try:
        result = pipeline.generate_from_text(
            req.prompt.strip(),
            seed=req.seed,
            mc_resolution=min(max(req.resolution, 64), 320),
            tier=req.tier,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _friendly(exc)
    return _pack(result)


@app.post("/api/image-to-3d")
async def image_to_3d(
    file: UploadFile = File(...), resolution: int = Form(256), tier: str = Form("mc")
):
    if tier not in ("mc", "ultra"):
        raise HTTPException(400, "tier must be 'mc' or 'ultra'")
    data = await file.read()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(413, "file too large")
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "not a readable image")
    # run in the threadpool so /api/progress stays responsive during the make
    try:
        _ensure_warm()
        result = await run_in_threadpool(
            pipeline.generate_from_image, image, min(max(resolution, 64), 320), tier
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _friendly(exc)
    return _pack(result)

"""Kilnform AI pipeline: translate -> SD-Turbo -> rembg -> TripoSR -> GLB.
All models are downloaded into backend/models/ once; generation is fully local.
"""
import io
import os
import sys
import time
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE / "models" / "hf"))
os.environ.setdefault("U2NET_HOME", str(BASE / "models" / "u2net"))
sys.path.insert(0, str(BASE))                       # let the torchmcubes shim win
sys.path.insert(1, str(BASE / "vendor" / "TripoSR"))

import numpy as np
import torch
import trimesh
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_cache = {}

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")


def _log(msg):
    print(f"[pipeline] {msg}", flush=True)


def get_translator():
    if "translator" not in _cache:
        _log("loading translator (opus-mt-tr-en)...")
        from transformers import pipeline as hf_pipeline
        _cache["translator"] = hf_pipeline(
            "translation", model="Helsinki-NLP/opus-mt-tr-en", device=-1
        )
    return _cache["translator"]


def translate_prompt(prompt: str) -> str:
    """Translate when Turkish characters are present; otherwise pass through."""
    if not (set(prompt) & TURKISH_CHARS):
        return prompt
    out = get_translator()(prompt, max_length=128)
    return out[0]["translation_text"]


def get_sd():
    if "sd" not in _cache:
        _log("loading SD-Turbo...")
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
        _cache["sd"] = pipe
    return _cache["sd"]


def get_tsr():
    if "tsr" not in _cache:
        _log("loading TripoSR...")
        from tsr.system import TSR
        model = TSR.from_pretrained(
            "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
        )
        model.renderer.set_chunk_size(8192)
        model.to(DEVICE)
        _cache["tsr"] = model
    return _cache["tsr"]


def get_rembg():
    if "rembg" not in _cache:
        _log("opening rembg session...")
        import rembg
        _cache["rembg"] = rembg.new_session()
    return _cache["rembg"]


STAGE = "idle"  # polled by GET /api/progress while a make runs


def _stage(name):
    global STAGE
    STAGE = name


def warmup():
    _stage("loading")
    try:
        get_translator()
        get_sd()
        get_tsr()
        get_rembg()
    finally:
        _stage("idle")
    return True


def text_to_image(prompt_en: str, seed: int | None = None) -> Image.Image:
    pipe = get_sd()
    gen = torch.Generator(device=DEVICE).manual_seed(seed) if seed is not None else None
    template = (
        f"a high quality 3d render of {prompt_en}, single object, centered, "
        f"full object in view, soft studio lighting, plain light gray background"
    )
    image = pipe(
        prompt=template,
        num_inference_steps=2,
        guidance_scale=0.0,
        height=512,
        width=512,
        generator=gen,
    ).images[0]
    return image


def image_to_mesh(image: Image.Image, mc_resolution: int = 256) -> tuple[bytes, Image.Image]:
    """PIL image -> (GLB bytes, processed preview image)."""
    from tsr.utils import remove_background, resize_foreground

    model = get_tsr()
    _stage("cutting")
    img = remove_background(image.convert("RGBA"), get_rembg())
    img = resize_foreground(img, 0.85)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    processed = Image.fromarray((arr * 255.0).astype(np.uint8))

    _stage("sculpting")
    with torch.no_grad():
        scene_codes = model([processed], device=DEVICE)
    _stage("extracting")
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]

    # make sure the skimage shim's winding faces outward
    if mesh.volume < 0:
        mesh.invert()
    # TripoSR builds z-up; glTF/three.js expect y-up
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    glb = mesh.export(file_type="glb")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()  # release activation memory between makes
    return glb, processed


def generate_from_text(prompt: str, seed: int | None = None, mc_resolution: int = 256):
    t0 = time.time()
    try:
        _stage("translating")
        prompt_en = translate_prompt(prompt)
        _log(f"prompt: {prompt!r} -> {prompt_en!r}")
        _stage("painting")
        image = text_to_image(prompt_en, seed)
        glb, preview = image_to_mesh(image, mc_resolution)
    finally:
        _stage("idle")
    return {
        "glb": glb,
        "preview": preview,
        "prompt_en": prompt_en,
        "seconds": round(time.time() - t0, 1),
    }


def generate_from_image(image: Image.Image, mc_resolution: int = 256):
    t0 = time.time()
    try:
        glb, preview = image_to_mesh(image, mc_resolution)
    finally:
        _stage("idle")
    return {
        "glb": glb,
        "preview": preview,
        "prompt_en": None,
        "seconds": round(time.time() - t0, 1),
    }


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

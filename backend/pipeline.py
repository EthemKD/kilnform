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
        # isnet cuts cleaner silhouettes than the default u2net; edge quality
        # feeds straight into mesh quality
        _cache["rembg"] = rembg.new_session("isnet-general-use")
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
    pipe.to(DEVICE)  # an Ultra make may have parked it on CPU
    gen = torch.Generator(device=DEVICE).manual_seed(seed) if seed is not None else None
    # The reference image drives both the mesh (TripoSR/Hunyuan) and its color
    # (TripoSR's forward pass), so wording it for a clean, evenly-lit, fully
    # framed subject with vivid color pays off twice. Even lighting / "no harsh
    # shadows" also keeps shadow from getting baked into the triplane color.
    template = (
        f"a high quality 3d render of {prompt_en}, single centered object, "
        f"full object fully in frame, vivid saturated colors, even soft studio "
        f"lighting, no harsh shadows, plain light gray background"
    )
    image = pipe(
        prompt=template,
        num_inference_steps=8,  # SD-Turbo is tuned for 1-4; 8 sharpens the
        guidance_scale=0.0,     # reference a little more, ~2-3s extra
        height=512,
        width=512,
        generator=gen,
    ).images[0]
    return image


def _ultra_mesh(rgba: Image.Image, processed: Image.Image, seed: int | None):
    """Ultra tier: Hunyuan3D-2mini sculpts the shape; TripoSR only paints it.

    Returns (mesh aligned to TripoSR's frame, TripoSR scene_codes for texbake).
    """
    import hunyuan

    _stage("sculpting-ultra")
    # 8GB budget: only one big model on the GPU at a time. SD-Turbo and
    # TripoSR both step aside so Hunyuan's octree decode gets full headroom —
    # measured 2-3x faster than leaving TripoSR resident (which forces the
    # decode to spill into shared GPU memory). Separately, octree_resolution
    # itself must stay <=448: at 512 the decode's *system RAM* footprint (not
    # VRAM) pushed the whole backend process past ~23GB and into pagefile
    # swap, which cost far more than the CPU<->GPU shuffle ever does.
    if DEVICE == "cuda":
        if "sd" in _cache:
            _cache["sd"].to("cpu")
        if "tsr" in _cache:
            _cache["tsr"].to("cpu")
        torch.cuda.empty_cache()
    mesh = hunyuan.generate(rgba, seed=seed)
    hunyuan.unload_gpu()
    mesh.apply_transform(HY_TO_TSR_ROT)

    # TripoSR forward pass on the same cutout: its triplane color field is what
    # texbake samples to paint the Hunyuan geometry
    model = get_tsr()
    model.to(DEVICE)  # no-op unless a failed make left it parked
    with torch.no_grad():
        scene_codes = model([processed], device=DEVICE)

    # Scale/offset differ per object (each model normalizes its own way), so
    # fit the rotated mesh onto a quick low-res TripoSR mesh of the same image.
    ref = model.extract_mesh(scene_codes, False, resolution=128)[0]
    src_min, src_max = mesh.bounds
    dst_min, dst_max = ref.bounds
    scale = (dst_max - dst_min) / np.maximum(src_max - src_min, 1e-9)
    fit = np.eye(4)
    fit[:3, :3] = np.diag(scale)
    fit[:3, 3] = dst_min - scale * src_min
    mesh.apply_transform(fit)
    return mesh, scene_codes


# Rotates Hunyuan3D-2mini output into TripoSR's frame: Hunyuan is y-up facing
# +z, TripoSR is z-up facing +x, so axes map (x,y,z) -> (z,x,y). Verified
# empirically by chamfer search + a visual cross-bake (gnome, seed 0).
HY_TO_TSR_ROT = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def image_to_mesh(
    image: Image.Image, mc_resolution: int = 256, tier: str = "mc", seed: int | None = None
) -> tuple[bytes, Image.Image]:
    """PIL image -> (GLB bytes, processed preview image)."""
    from tsr.utils import remove_background, resize_foreground

    model = get_tsr()
    _stage("cutting")
    img = remove_background(image.convert("RGBA"), get_rembg())
    img = resize_foreground(img, 0.85)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    processed = Image.fromarray((arr * 255.0).astype(np.uint8))

    if tier == "ultra":
        mesh, scene_codes = _ultra_mesh(img, processed, seed)
        face_budget = 100000  # Ultra's point is geometry; keep more of it
    else:
        _stage("sculpting")
        model.to(DEVICE)  # no-op unless a failed Ultra make left it parked
        with torch.no_grad():
            scene_codes = model([processed], device=DEVICE)
        _stage("extracting")
        mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]

        # make sure the skimage shim's winding faces outward
        if mesh.volume < 0:
            mesh.invert()

        # soften marching-cubes staircase artifacts; vertex attributes ride along
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=5)
        face_budget = 50000

    _stage("texturing")
    try:
        import texbake
        out_mesh, _ = texbake.bake(mesh, model, scene_codes[0], face_budget=face_budget)
    except Exception as exc:  # e.g. no GL context on a headless box
        # Don't ship a blank gray model (Ultra's Hunyuan mesh has no vertex
        # color of its own): sample TripoSR's triplane color straight onto the
        # vertices instead. Same color source as the bake, minus the UV atlas.
        _log(f"texture bake failed ({exc!r}); falling back to per-vertex triplane color")
        try:
            import texbake
            pts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                vc = model.renderer.query_triplane(model.decoder, pts, scene_codes[0])["color"]
            vc = texbake._lift_color(vc.float().cpu().numpy().reshape(1, -1, 3))[0]
            mesh.visual.vertex_colors = (vc * 255.0).astype(np.uint8)
        except Exception as exc2:
            _log(f"vertex-color fallback also failed ({exc2!r}); shipping untextured")
        out_mesh = mesh

    # TripoSR builds z-up; glTF/three.js expect y-up
    out_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    glb = out_mesh.export(file_type="glb")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()  # release activation memory between makes
    return glb, processed


def generate_from_text(
    prompt: str, seed: int | None = None, mc_resolution: int = 256, tier: str = "mc"
):
    t0 = time.time()
    try:
        _stage("translating")
        prompt_en = translate_prompt(prompt)
        _log(f"prompt: {prompt!r} -> {prompt_en!r}")
        _stage("painting")
        image = text_to_image(prompt_en, seed)
        glb, preview = image_to_mesh(image, mc_resolution, tier=tier, seed=seed)
    finally:
        _stage("idle")
    return {
        "glb": glb,
        "preview": preview,
        "prompt_en": prompt_en,
        "seconds": round(time.time() - t0, 1),
    }


def generate_from_image(image: Image.Image, mc_resolution: int = 256, tier: str = "mc"):
    t0 = time.time()
    try:
        glb, preview = image_to_mesh(image, mc_resolution, tier=tier)
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

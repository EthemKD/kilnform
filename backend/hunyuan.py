"""Hunyuan3D-2mini shape generation — the "Ultra" detail tier.

Only the shapegen half of the vendored Hunyuan3D-2 repo is used (no
custom_rasterizer/texgen, nothing to compile). Weights land in models/hf via
HF_HOME like every other model. License: Tencent Hunyuan 3D 2.0 Community
License — see README's license section for the territory note.
"""
import pathlib
import sys

import torch

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(1, str(BASE / "vendor" / "Hunyuan3D-2"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_cache = {}


def get_pipeline():
    if "hy" not in _cache:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mini",
            subfolder="hunyuan3d-dit-v2-mini",
            device=DEVICE,
        )
        # FlashVDM cuts VAE volume decoding 2-3x at equal quality
        pipe.enable_flashvdm(topk_mode="merge")
        _cache["hy"] = pipe
    return _cache["hy"]


def unload_gpu():
    """Park the pipeline in system RAM. 8GB cards can't hold Hunyuan and
    TripoSR at once; the ~2s round trip beats spilling into shared memory
    (which slowed TripoSR 41s vs 16s when everything stayed resident)."""
    if "hy" in _cache and DEVICE == "cuda":
        _cache["hy"].to("cpu")
        torch.cuda.empty_cache()


def generate(rgba_image, seed=None, steps=30, octree_resolution=380, guidance=5.0):
    """RGBA cutout -> cleaned trimesh in Hunyuan's own frame."""
    from hy3dgen.shapegen import DegenerateFaceRemover, FloaterRemover

    pipe = get_pipeline()
    pipe.to(DEVICE)  # may be parked on CPU from a previous make
    gen = (
        torch.Generator(device=DEVICE).manual_seed(seed) if seed is not None else None
    )
    mesh = pipe(
        image=rgba_image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        octree_resolution=octree_resolution,
        generator=gen,
        enable_pbar=False,
    )[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    if mesh.volume < 0:  # keep faces wound outward for GL backface culling
        mesh.invert()
    return mesh

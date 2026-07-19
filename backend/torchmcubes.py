"""torchmcubes shim — replaces the original package (which needs a C++ build on
Windows) with skimage.measure.marching_cubes. TripoSR's tsr/models/isosurface.py
finds this module via `from torchmcubes import marching_cubes` because backend/
comes first on sys.path.

Note: isosurface.py flips the result with v_pos[..., [2,1,0]]; skimage returns
(dim0,dim1,dim2) order, so we pre-flip here and the double flip cancels out.
"""
import numpy as np
import torch
from skimage import measure


def marching_cubes(volume, threshold=0.0):
    if isinstance(volume, torch.Tensor):
        vol = volume.detach().cpu().numpy()
    else:
        vol = np.asarray(volume)
    verts, faces, _normals, _values = measure.marching_cubes(vol.astype(np.float32), level=float(threshold))
    verts = np.ascontiguousarray(verts[:, ::-1])  # (d0,d1,d2) -> (d2,d1,d0)
    return (
        torch.from_numpy(verts.astype(np.float32)),
        torch.from_numpy(faces.astype(np.int64)),
    )

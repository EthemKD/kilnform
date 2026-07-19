"""UV texture baking for TripoSR meshes.

Adapted from TripoSR's tsr/bake_texture.py (MIT, VAST-AI-Research) with three
changes: the moderngl context and shader programs are created once and reused
(the original leaked a GL context per bake), per-bake GL objects are released,
and the triplane color query runs on the model's device instead of the CPU.
"""
import moderngl
import numpy as np
import torch
import trimesh
import xatlas
from PIL import Image

BASIC_VERTEX_SHADER = """
    #version 330
    in vec2 in_uv;
    in vec3 in_pos;
    out vec3 v_pos;
    void main() {
        v_pos = in_pos;
        gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
    }
"""
BASIC_FRAGMENT_SHADER = """
    #version 330
    in vec3 v_pos;
    out vec4 o_col;
    void main() {
        o_col = vec4(v_pos, 1.0);
    }
"""
GS_VERTEX_SHADER = """
    #version 330
    in vec2 in_uv;
    in vec3 in_pos;
    out vec3 vg_pos;
    void main() {
        vg_pos = in_pos;
        gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
    }
"""
GS_GEOMETRY_SHADER = """
    #version 330
    uniform float u_resolution;
    uniform float u_dilation;
    layout (triangles) in;
    layout (triangle_strip, max_vertices = 12) out;
    in vec3 vg_pos[];
    out vec3 vf_pos;
    void lineSegment(int aidx, int bidx) {
        vec2 a = gl_in[aidx].gl_Position.xy;
        vec2 b = gl_in[bidx].gl_Position.xy;
        vec3 aCol = vg_pos[aidx];
        vec3 bCol = vg_pos[bidx];

        vec2 dir = normalize((b - a) * u_resolution);
        vec2 offset = vec2(-dir.y, dir.x) * u_dilation / u_resolution;

        gl_Position = vec4(a + offset, 0.0, 1.0);
        vf_pos = aCol;
        EmitVertex();
        gl_Position = vec4(a - offset, 0.0, 1.0);
        vf_pos = aCol;
        EmitVertex();
        gl_Position = vec4(b + offset, 0.0, 1.0);
        vf_pos = bCol;
        EmitVertex();
        gl_Position = vec4(b - offset, 0.0, 1.0);
        vf_pos = bCol;
        EmitVertex();
    }
    void main() {
        lineSegment(0, 1);
        lineSegment(1, 2);
        lineSegment(2, 0);
        EndPrimitive();
    }
"""
GS_FRAGMENT_SHADER = """
    #version 330
    in vec3 vf_pos;
    out vec4 o_col;
    void main() {
        o_col = vec4(vf_pos, 1.0);
    }
"""

_gl = {}


def _programs():
    if "ctx" not in _gl:
        ctx = moderngl.create_context(standalone=True)
        _gl["ctx"] = ctx
        _gl["basic"] = ctx.program(
            vertex_shader=BASIC_VERTEX_SHADER, fragment_shader=BASIC_FRAGMENT_SHADER
        )
        _gl["gs"] = ctx.program(
            vertex_shader=GS_VERTEX_SHADER,
            geometry_shader=GS_GEOMETRY_SHADER,
            fragment_shader=GS_FRAGMENT_SHADER,
        )
    return _gl["ctx"], _gl["basic"], _gl["gs"]


def _rasterize_positions(mesh, vmapping, indices, uvs, resolution, padding):
    """Render world positions into atlas space; rows come back bottom-first."""
    ctx, basic_prog, gs_prog = _programs()
    vbo_uvs = ctx.buffer(uvs.flatten().astype("f4"))
    vbo_pos = ctx.buffer(mesh.vertices[vmapping].flatten().astype("f4"))
    ibo = ctx.buffer(indices.flatten().astype("i4"))
    vao_content = [
        vbo_uvs.bind("in_uv", layout="2f"),
        vbo_pos.bind("in_pos", layout="3f"),
    ]
    basic_vao = ctx.vertex_array(basic_prog, vao_content, ibo)
    gs_vao = ctx.vertex_array(gs_prog, vao_content, ibo)
    tex = ctx.texture((resolution, resolution), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[tex])
    try:
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        gs_prog["u_resolution"].value = resolution
        gs_prog["u_dilation"].value = padding
        gs_vao.render()
        basic_vao.render()
        raw = fbo.read(components=4, dtype="f4")
    finally:
        for obj in (basic_vao, gs_vao, fbo, tex, vbo_uvs, vbo_pos, ibo):
            obj.release()
    return np.frombuffer(raw, dtype="f4").reshape(resolution, resolution, 4)


def bake(mesh, model, scene_code, texture_resolution=1024, face_budget=50000):
    """Mesh + triplane -> UV-textured copy of the mesh (plus its texture image).

    The input mesh is left untouched; detail beyond face_budget moves into the
    texture (xatlas unwrap time scales with face count).
    """
    if len(mesh.faces) > face_budget:
        mesh = mesh.simplify_quadric_decimation(face_count=face_budget)

    padding = round(max(2, texture_resolution / 256))
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces)
    pack = xatlas.PackOptions()
    pack.resolution = texture_resolution
    pack.padding = padding
    pack.bilinear = True
    atlas.generate(pack_options=pack)
    vmapping, indices, uvs = atlas[0]

    positions_texture = _rasterize_positions(
        mesh, vmapping, indices, uvs, texture_resolution, padding
    )
    flat = positions_texture.reshape(-1, 4)
    positions = torch.tensor(flat[:, :3], device=scene_code.device)
    with torch.no_grad():
        queried = model.renderer.query_triplane(model.decoder, positions, scene_code)
    colors = (
        queried["color"].float().cpu().numpy()
        .reshape(texture_resolution, texture_resolution, 3)
    )
    mask = flat[:, 3].reshape(texture_resolution, texture_resolution) == 0.0
    colors[mask] = 0.5  # neutral fill outside the atlas islands

    # GL readback rows are bottom-first (row 0 = v0); flipping puts v=1 at the
    # image top, which is what an image paired with bottom-origin uvs needs
    texture = Image.fromarray((colors * 255.0).astype(np.uint8)).transpose(
        Image.FLIP_TOP_BOTTOM
    )
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture, metallicFactor=0.0, roughnessFactor=1.0
    )
    textured = trimesh.Trimesh(
        vertices=mesh.vertices[vmapping],
        faces=indices,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        process=False,
    )
    return textured, texture

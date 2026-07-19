"""Standalone pipeline test: downloads/loads the models, then makes a GLB from a Turkish prompt."""
import io
import time

import pipeline


def main():
    t0 = time.time()
    print("== warmup (first run downloads the models) ==", flush=True)
    pipeline.warmup()
    print(f"warmup done: {time.time()-t0:.0f}s", flush=True)

    result = pipeline.generate_from_text("çeşme", seed=42)
    glb = result["glb"]
    print(f"generation: {result['seconds']}s, translation: {result['prompt_en']!r}", flush=True)
    print(f"glb size: {len(glb)} bytes, magic: {glb[:4]!r}", flush=True)

    assert glb[:4] == b"glTF", "bad GLB magic!"
    assert len(glb) > 100_000, "GLB suspiciously small!"

    import trimesh
    mesh = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
    print(f"triangles: {len(mesh.faces)}, vertices: {len(mesh.vertices)}", flush=True)
    assert len(mesh.faces) > 5000, "too few triangles!"

    with open("test_output.glb", "wb") as f:
        f.write(glb)
    result["preview"].save("test_output_preview.png")
    print("OK — wrote test_output.glb and test_output_preview.png", flush=True)


if __name__ == "__main__":
    main()

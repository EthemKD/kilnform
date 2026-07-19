# ◆ Kilnform

A local 3D workshop. Type a prompt or hand over an image; a 3D model gets made
**on your own machine** — viewed, styled, and exported without a single byte
leaving localhost.

![Kilnform: "garden gnome with red hat" sculpted and texture-baked locally](docs/screenshot.png)
<p align="center"><i>"garden gnome with red hat" → 50,000 triangles + a baked 1024² UV texture in 16 seconds on an RTX 5060 Laptop — entirely offline. The shelf below holds earlier makes: a teapot, a treasure chest, two fountains.</i></p>

Two engines share one bench:

- **AI · realistic** — a fully local pipeline: your prompt is translated if needed
  (opus-mt, CPU), painted into a reference image (SD-Turbo), cut out (rembg/isnet),
  sculpted into a real mesh (TripoSR) on your GPU, then smoothed and baked into a
  UV-textured GLB. A prompt like `çeşme` becomes an actual tiered fountain — crisp
  1024² texture included — in well under half a minute.
- **Instant** — a deterministic procedural generator. It parses Turkish/English
  keywords (house, tree, car, robot, rocket… plus color/size/material adjectives)
  and builds parametric models with seeded variations. No GPU, no wait, works on
  any machine.

Everything downstream is engine-agnostic: orbit viewer with studio lighting,
material editing, stylization (voxelize / low-poly / toon), transforms, an
IndexedDB library shelf, and GLB / OBJ / STL export (STL is 3D-print ready).

Prompts can be Turkish or English — the UI is English, the understanding is both.

## Privacy, as a design rule

- Both servers bind to `127.0.0.1` only.
- Model weights are downloaded once at setup; after that the whole thing runs offline.
- Images, prompts, and generated meshes never touch the network. The top bar has a
  live sentinel that flags any request leaving localhost — it should never trigger.

## System requirements

| | Instant engine | AI engine |
|---|---|---|
| OS | any modern browser | Windows 10/11 (tested); Linux should work with minor script changes |
| Node.js | 18+ | 18+ |
| Python | — | 3.12 (installed automatically via `uv`) |
| GPU | not needed | NVIDIA, ~6.5GB free VRAM (8GB card recommended); Blackwell (RTX 50xx) needs the CUDA 12.8 wheels the setup installs. CPU fallback works but is slow (~1–2 min/model) |
| RAM | any | 16GB+ recommended (backend holds ~4–5GB) |
| Disk | ~200MB | ~12GB (PyTorch + model weights) |

Measured on an RTX 5060 Laptop (8GB): a make is a 5–16s burst depending on the
detail tier (Fine bakes the 1024² texture), ~45W / 64°C peak, ~6.5GB VRAM peak,
idle between requests. Laptop-friendly. One Windows gotcha handled in code: a
minimized backend console can be put in EcoQoS ("efficiency mode"), which made
generation 4x slower — the backend opts itself out at startup.

## Setup

```bat
npm install            # web UI dependencies
backend\setup.bat      # one-time: Python env, PyTorch cu128, TripoSR, model weights (~5GB download)
```

Skipping `backend\setup.bat` is fine — the app runs in Instant mode without it.

## Run

```bat
start.bat              # starts backend + web UI, opens http://127.0.0.1:5173
```

## How the AI pipeline stays Windows-friendly

TripoSR normally requires `torchmcubes`, which needs a C++/CUDA build. Kilnform
ships a small shim ([backend/torchmcubes.py](backend/torchmcubes.py)) that
provides the same interface via `skimage.measure.marching_cubes`, so nothing has
to compile. Mesh winding is normalized and the result is rotated z-up → y-up
before export. Texture baking ([backend/texbake.py](backend/texbake.py)) is
adapted from TripoSR's bake code with two fixes: one reused GL context instead
of a fresh (leaked) one per make, and triplane color queries on the GPU.

## Licenses of the parts

Kilnform's own code is yours to license as you wish. The pieces it stands on:

- [three.js](https://threejs.org) — MIT
- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (code & weights) — MIT
- [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) weights — **Stability AI
  non-commercial research license**. Weights are downloaded by each user at setup,
  not redistributed here. If you need commercial use, swap the model id in
  `backend/pipeline.py` for a permissively licensed one.
- Helsinki-NLP opus-mt-tr-en — CC-BY 4.0 · rembg/u2net — Apache-2.0

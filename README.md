🐣 Please follow me for new updates: https://x.com/camenduru <br />
🔥 Please join our discord server: https://discord.gg/k5BwmmvJJU <br />
🥳 Please become my sponsor: https://github.com/sponsors/camenduru <br />
🍞 TostUI repo: https://github.com/camenduru/TostUI

#### 🍞 Tost AI - Sound Effect Studio

A web app for **MOSS-SoundEffect v2.0** that exposes every capability the
model has — text-to-sound in English and Chinese, duration up to 30 s with the
training-time duration tag, flow-matching sampling control (steps, CFG,
sigma shift, seed), an optional negative prompt, and batch generation —
behind one flat, light/dark interface. Every take is written to an output
folder and listed in the UI.

```
tostai-sound-effect-studio/
├── server.py              FastAPI app: pipeline loader, generation, output shelf
├── requirements.txt       Web-app dependencies (the model's deps come from the model env)
├── Dockerfile             self-contained image: clones the code, pulls the weights
├── docker-entrypoint.sh   one process: the studio IS the model server
├── docker_selfcheck.py    build-time proof that the image is complete
├── smoke_generate.py      runtime smoke test against a live server
├── outputs/               every finished take: <name>.wav + <name>.json
└── static/
    ├── index.html         interface, with the pre-paint theme script
    ├── styles.css         both palettes, one set of variable names
    ├── app.js             Web Audio player, visualiser, live progress, output shelf
    ├── logo.png           the TostAI mark from the toolbar's favicon
    └── favicon.ico        same mark, 16/32/48 px
```

## Run it

The app has its own virtualenv, so its four web dependencies never touch the
interpreter the model runs on. Locally, the model environment — the sibling
`MOSS-TTS` checkout with its `moss_soundeffect_v2/.venv` — is layered under
this venv automatically:

```bash
cd tostai-sound-effect-studio
python -m venv .venv                                   # once (python 3.12)
.venv/Scripts/python.exe -m pip install -r requirements.txt   # once (POSIX: .venv/bin/python)
.venv/Scripts/python.exe server.py --port 8000
```

Open <http://127.0.0.1:8000>.

### Give it a model

On first generate, the server loads `MossSoundEffectPipeline` from
`SOUNDEFFECT_MODEL_DIR` (default: the public HuggingFace repo
`OpenMOSS-Team/MOSS-SoundEffect-v2.0`, ~10.5 GB downloaded once into the HF
cache) onto CUDA. Point it at the local checkout instead:

```bash
SOUNDEFFECT_MODEL_DIR=../MOSS-SoundEffect-v2.0 .venv/Scripts/python.exe server.py --port 8000
```

The header badge flips from **demo audio (model offline)** to a green **model
online** dot once the pipeline is ready. The first load takes a couple of
minutes; the status endpoint reports `loading` the whole time, and the badge
tooltip shows what is happening.

The DiT is wrapped with `torch.compile` + Triton CUDA Graphs upstream; this
app sets `TORCHDYNAMO_DISABLE=1` by default (as the model's own scripts do) so
the first generation cannot die inside a Triton compile on an unsupported GPU.
Export `TORCHDYNAMO_DISABLE=` (empty) to let torch.compile run if your GPU
supports it.

### No GPU? It still works

If the model cannot load the app **falls back to a clearly-labelled demo
synth** so you can explore the whole interface; demo takes are badged `demo`
in the shelf and the player (`X-SoundEffect-Demo: 1`). Force it with `--demo`,
or make an unloadable model a hard 503 with `--no-demo-fallback`.

## What the app covers

| MOSS-SoundEffect v2.0 capability | Where |
| --- | --- |
| **Text-to-sound (EN + 中文)** | Prompt field + EN/中文 toggle swaps quick prompts and presets |
| **Duration 0.5–30 s** | Duration slider; `duration: Xs` appended exactly as at training time |
| **Duration tag on/off** | `append_duration_suffix` switch in Sampling controls |
| **num_inference_steps** | Sampling controls (10–150, default 100) |
| **cfg_scale** | Sampling controls (1–8, default 4.0); 1.0 disables CFG |
| **sigma_shift** | Sampling controls (0–10, default 5.0) |
| **seed** | Sampling controls, with randomiser; same seed + settings = same take |
| **negative_prompt** | Negative prompt field + presets (steered by CFG) |
| **Batch prompts** | "switch to batch" — one caption per line, one take per caption |
| **Step progress** | Live `step n / N` progress bar with ETA while diffusion runs |
| **48 kHz mono WAV** | Player, per-take download, and the output folder |

## The output folder

Every finished take is written to `outputs/` as a WAV **plus a JSON sidecar**
holding the prompt(s), negative prompt, duration, steps, CFG, sigma shift,
seed, duration-tag flag, demo flag and timing. The shelf in the UI is a *view
of that folder* — it is re-read from disk, not mirrored in the browser — so
takes survive a page reload, a server restart and a container restart, and you
can also just open the folder.

* WAV is mono 48 kHz 16-bit, exactly what the model emits.
* `<name>` is `sfx-<timestamp>-<mode>-seed<seed>.wav`.
* Override the location with `SOUNDEFFECT_OUTPUTS_DIR` (the image mounts it).
* Delete a take with the ✕ in the shelf, or delete the pair by hand. A sidecar
  whose WAV has gone is skipped rather than shown as broken.

## Light and dark

Both themes use the same CSS variable names, so no rule knows which is active:
`:root` holds the light palette and `html[data-theme="dark"]` overrides it, and
`color-scheme` hands the browser's own controls over to the same decision.

The choice is made by an inline script in `index.html` **before the first
paint** — otherwise a dark page flashes the light palette and repaints. A
stored choice in `localStorage` (`tostai.sfx.theme`) wins; with none, the OS
decides. The button names the theme it switches *to* ("Dark" on a light page),
and the canvas visualiser re-reads `--acc` whenever the theme changes, since a
canvas cannot use CSS variables.

The palette, the component shapes and the app icon are taken from TostAI
Sprite Studio (`ui.html`) — the same reference the Voice Studio uses: the same
`--bd/--tx/--mut/--acc/--accbg/--acctext/--acch` names, the same flat
1px-bordered surfaces, the same `+`/`–` `<details>` panels, and the same mark
— `static/logo.png` is that page's inline header logo and `static/favicon.ico`
is that project's toolbar icon. To retheme, edit the two variable blocks at
the top of `styles.css` and nothing else.

## Docker

Pull the published image — no build, no model download, no tokens:

```bash
docker pull camenduru/tostai-sound-effect-studio
docker run --rm --gpus all -p 8000:8000 camenduru/tostai-sound-effect-studio
```

then open <http://127.0.0.1:8000>.

One self-contained image: it clones the inference code, clones the studio from
its own GitHub repo, and downloads the weights during the build, so it needs
no local model, and it runs both from one process. To build it yourself:

```bash
cd tostai-sound-effect-studio
docker build --build-arg CACHEBUST=$(date +%s) -t camenduru/tostai-sound-effect-studio .
docker run --rm --gpus all -p 8000:8000 camenduru/tostai-sound-effect-studio
```

All three source repositories are public, so the build needs no credentials.
`HF_TOKEN` is accepted as an optional secret mount (lifts the HuggingFace rate
limit on a busy build farm). See [Configuration](#configuration) for the
`set -a; . ./.env` step.

| | |
| --- | --- |
| Inference code | `git clone https://github.com/OpenMOSS/MOSS-TTS` → `/app/MOSS-TTS` |
| Studio | `git clone https://github.com/camenduru/TostAI-Sound-Effect-Studio` → `/app/tostai-sound-effect-studio` |
| Weights | `OpenMOSS-Team/MOSS-SoundEffect-v2.0` (~10.5 GB) → `/app/MOSS-SoundEffect-v2.0` |
| Studio port | 8000 (UI + model API in one process) |
| User | `camenduru` (non-root) |
| Python | `/opt/venv` (3.12), one venv for model and studio |

The build context is **this directory**, not the repo root — `.dockerignore`
keeps it to about 1 kB by excluding `.venv/`, caches and local audio. The
studio's files come from the clone, not the context; only `docker_selfcheck.py`
is copied from the context.

Notes worth knowing before you build:

* **No credentials needed.** Every repository this build reads is public, so
  plain `git clone` and an unauthenticated weight download are all it takes.
  `HF_TOKEN` is optional; when supplied it arrives as a secret mount
  (`--secret id=hf_token,env=HF_TOKEN`) — never as `ARG` or `ENV`, so it stays
  out of `docker history`.
* **`CACHEBUST` is not optional in practice.** BuildKit caches the studio clone
  under a key that ignores what the branch points at now, so without
  `--build-arg CACHEBUST=$(date +%s)` a rebuild silently re-serves the first
  snapshot. The flag only invalidates the clone and the cheap layers after it.
* **No CUDA toolkit is installed.** The pip torch wheels carry their own CUDA
  runtime; only the host driver is required, which is why `--gpus all` is the
  whole GPU story.
* The build runs `docker_selfcheck.py`, which asserts the pipeline imports,
  every weights directory really arrived (a truncated download is the failure
  this catches), torch is a CUDA build, and the studio answers its own routes.
  It does **not** load the model — a build must not depend on a GPU.
* `SOUNDEFFECT_SERVE_MODEL=0` makes the container UI-only (demo fallback),
  and `SOUNDEFFECT_MODEL_DIR` can point at a mounted checkpoint instead.

### Updating a running container

The **Update** button in the header pulls the latest studio source from
`camenduru/TostAI-Sound-Effect-Studio` and restarts the server in place, with
no rebuild and no token — the repository is public. This is a dev convenience —
the files it writes live in the container and die with it. The durable path is
still a rebuild.

## Configuration

`.env` holds one **optional credential** — `HF_TOKEN` — and nothing else. It
is gitignored and excluded from the Docker build context, so nothing in it is
committed or uploaded to the builder.

```bash
set -a; . ./.env; set +a        # `set -a` is required: it marks values for export
docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg CACHEBUST=$(date +%s) \
  -t camenduru/tostai-sound-effect-studio .
```

Or skip the secret entirely — a plain build works:

```bash
docker build --build-arg CACHEBUST=$(date +%s) -t camenduru/tostai-sound-effect-studio .
```

### Publishing to Docker Hub (`camenduru/tostai-sound-effect-studio`)

```bash
docker login
docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg CACHEBUST=$(date +%s) \
  -t camenduru/tostai-sound-effect-studio:latest .
docker push camenduru/tostai-sound-effect-studio:latest
# optional version tag:
# docker tag camenduru/tostai-sound-effect-studio:latest camenduru/tostai-sound-effect-studio:<version>
# docker push camenduru/tostai-sound-effect-studio:<version>
```

The token arrives as a secret mount, never as `ARG` or `ENV`, so it stays out
of `docker history` and the image config. Each assignment is guarded as
`NAME=${NAME:-}`, so a value already in your environment wins — keep the guard
if you edit the file.

The app's own behaviour is set with **ordinary environment variables**, which
belong in your shell or in `docker run -e` rather than in a credentials file:

| Variable | Default | Effect |
| --- | --- | --- |
| `SOUNDEFFECT_MODEL_DIR` | `OpenMOSS-Team/MOSS-SoundEffect-v2.0` | Checkpoint dir or HF repo id to load |
| `SOUNDEFFECT_DEVICE` | `auto` | `cuda`, `cpu` or `auto` |
| `SOUNDEFFECT_OUTPUTS_DIR` | `./outputs` | Where takes are written; point it at a volume |
| `SOUNDEFFECT_SERVE_MODEL` | `1` | `0` runs the UI alone (demo fallback) |
| `SOUNDEFFECT_STUDIO_PORT` | `8000` | UI port (inside the container) |
| `SOUNDEFFECT_REV` | `main` | Checkpoint revision baked into the image (build-time) |
| `MOSS_CODE_DIR` | sibling `MOSS-TTS` / `/app/MOSS-TTS` | Where `moss_soundeffect_v2` is imported from |
| `MOSS_VENV_DIR` | `MOSS_CODE_DIR/moss_soundeffect_v2/.venv` | Local layered model venv (ignored in Docker) |
| `TOSTAI_APP_REPO` | `camenduru/TostAI-Sound-Effect-Studio` | Repo the Update button pulls from |
| `TORCHDYNAMO_DISABLE` | `1` | Keep the DiT's torch.compile/Triton wrapper off |

An **empty** value is treated as unset by the studio, so `SOUNDEFFECT_OUTPUTS_DIR=`
keeps the default folder.

## HTTP surface

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | The app |
| `GET` | `/api/catalog` | Parameter ranges, presets, parameter docs, model facts |
| `GET` | `/api/status` | Engine state (`idle` / `loading` / `ready` / `failed`) + demo fallback |
| `POST` | `/api/generate` | Form in, **WAV** out (single prompt) or JSON batch; saves the take(s) |
| `GET` | `/api/generate/progress` | Live step counter and ETA while diffusion runs |
| `GET` | `/api/outputs` | Every saved take, newest first, with its metadata |
| `GET` | `/api/outputs/{name}` | One saved take as a WAV |
| `DELETE` | `/api/outputs/{name}` | Remove a take (audio + sidecar) |
| `GET` | `/api/update` | The installed studio revision (for the Update dialog) |
| `POST` | `/api/update` | Pull latest source, then restart |

`POST /api/generate` accepts `prompts` (one caption per line; a single line is
the ordinary path), `negative_prompt`, `seconds`, `steps`, `cfg_scale`,
`sigma_shift`, `seed`, `duration_tag` and `save`, mirroring
`MossSoundEffectPipeline.__call__`.

## License

The app code here is yours to use. MOSS-SoundEffect v2.0 model weights and
code are **Apache-2.0** — see the
[model card](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0) and
the [MOSS-TTS repository](https://github.com/OpenMOSS/MOSS-TTS).

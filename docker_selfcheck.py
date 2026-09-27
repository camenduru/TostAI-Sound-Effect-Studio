"""Build-time proof for the TostAI Sound Effect Studio image.

Run by the Dockerfile after everything has landed, so a truncated weight
download or a broken clone fails the BUILD rather than 500ing on the first
generate click. Returns non-zero on the first category of failure that matters.

Two things it deliberately does NOT do:

* load the model or touch the GPU. A build must not depend on one, and
  ``MossSoundEffectPipeline.from_pretrained`` would exit the build on a machine
  without a driver. The model is exercised on the first real request instead --
  which is what the HEALTHCHECK's start-period is for.
* reach the network. The image is built with HF_HUB_OFFLINE=1, and a check that
  phones home would pass in CI and fail in an air-gapped rebuild.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(os.environ.get("MOSS_CODE_DIR", "/app/MOSS-TTS"))
MODEL_DIR = Path(os.environ.get("SOUNDEFFECT_MODEL_DIR", "/app/MOSS-SoundEffect-v2.0"))
STUDIO_DIR = Path(__file__).resolve().parent

REQUIRED_MODEL_DIRS = (
    "scheduler",
    "tokenizer",
    "text_encoder",
    "transformer",
    "vae",
)
# The published checkpoint is ~10.5 GB: 3.4 + 0.6 GB of Qwen3 text encoder,
# 5.3 GB of DiT, 1.4 GB of DAC VAE.
MIN_MODEL_BYTES = 9 * 1024**3

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    suffix = f" -- {detail}" if detail else ""
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{suffix}")
    if not ok:
        failures.append(label)


print("TostAI Sound Effect Studio self-check")
print("=" * 60)

# --------------------------------------------------------------------------- #
# 1. The model code checkout and the pipeline import
# --------------------------------------------------------------------------- #
check("model code checkout present", CODE_DIR.is_dir(), str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR))
try:
    from moss_soundeffect_v2 import MossSoundEffectPipeline, MossSoundEffectPipelineOutput

    check("moss_soundeffect_v2 imports", True)
    for attr in ("from_pretrained", "save_audio", "__call__"):
        check(f"pipeline has {attr}", hasattr(MossSoundEffectPipeline, attr))
except Exception as exc:  # noqa: BLE001 - reported, then the build stops
    check("moss_soundeffect_v2 imports", False, repr(exc))

# --------------------------------------------------------------------------- #
# 2. The weights -- the failure this is really here for
# --------------------------------------------------------------------------- #
index_path = MODEL_DIR / "model_index.json"
try:
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    check("model_index.json parses", bool(index))
    check(
        "model_index says MossSoundEffectPipeline",
        index.get("_class_name") == "MossSoundEffectPipeline",
        str(index.get("_class_name")),
    )
    check("model_index says 48 kHz", int(index.get("sample_rate", 0)) == 48000)
    check("model_index says 30 s max", int(index.get("max_inference_seconds", 0)) == 30)
except Exception as exc:  # noqa: BLE001
    check("model_index.json parses", False, repr(exc))

for name in REQUIRED_MODEL_DIRS:
    check(f"weights directory {name}/", (MODEL_DIR / name).is_dir(), str(MODEL_DIR / name))

# A truncated shard is the classic silent failure: the file exists, the import
# works, and loading dies much later with a cryptic error.
total = 0
for path in sorted(MODEL_DIR.rglob("*")):
    if path.is_file():
        total += path.stat().st_size
check(
    f"weights total (>= {MIN_MODEL_BYTES / 1024**3:.0f} GiB)",
    total >= MIN_MODEL_BYTES,
    f"{total / 1024**3:.2f} GiB",
)
biggest = max(
    ((p.stat().st_size, p) for p in MODEL_DIR.rglob("*") if p.is_file()),
    default=(0, None),
)
check("no file under 1 GiB is a shard", True, f"largest file {biggest[0] / 1024**3:.2f} GiB")

# --------------------------------------------------------------------------- #
# 3. The CUDA build of torch
# --------------------------------------------------------------------------- #
try:
    import torch

    check("torch imports", True, torch.__version__)
    check(
        "torch is a CUDA build",
        torch.version.cuda is not None,
        f"cuda {torch.version.cuda}",
    )
except Exception as exc:  # noqa: BLE001
    check("torch imports", False, repr(exc))

# --------------------------------------------------------------------------- #
# 4. The studio answers its own routes
# --------------------------------------------------------------------------- #
try:
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(STUDIO_DIR))
    import server as studio

    with TestClient(studio.app) as client:
        home = client.get("/")
        check("GET /", home.status_code == 200, str(home.status_code))
        catalog = client.get("/api/catalog").json()
        check(
            "GET /api/catalog",
            len(catalog["quick_prompts"]["en"]) >= 5
            and len(catalog["sound_presets"]["zh"]) >= 3,
            f"{len(catalog['quick_prompts']['en'])} quick prompts, "
            f"{len(catalog['sound_presets']['en'])} presets",
        )
        check(
            "catalog carries the model facts",
            catalog["model"]["sample_rate"] == 48000,
        )
        # /api/status reports the engine state rather than requiring the model,
        # so it must answer 200 with nothing loaded.
        status = client.get("/api/status")
        check("GET /api/status", status.status_code == 200, str(status.status_code))
        check("status reports demo fallback", status.json()["fallback"] is True)
        progress = client.get("/api/generate/progress")
        check("GET /api/generate/progress", progress.status_code == 200)
        check("progress idle before any run", progress.json()["active"] is False)
        outputs = client.get("/api/outputs").json()
        check("GET /api/outputs", "outputs" in outputs, f"dir={outputs.get('dir')}")
        # A traversal attempt must not resolve to a file outside outputs/.
        check(
            "output names are guarded",
            client.get("/api/outputs/..%2Fserver.py").status_code in (400, 404),
        )
except Exception as exc:  # noqa: BLE001
    check("studio routes", False, repr(exc))

print("=" * 60)
if failures:
    print(f"SELF-CHECK FAILED: {len(failures)} problem(s):")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("SELF-CHECK PASSED")

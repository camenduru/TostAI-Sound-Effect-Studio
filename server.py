"""TostAI Sound Effect Studio -- a web app for MOSS-SoundEffect v2.0.

Wraps every capability of the ``MossSoundEffectPipeline`` behind one small HTTP
surface:

* text-to-sound-effect from a natural-language caption (EN or ZH -- the model
  was trained on both)
* duration control, 0.5 .. 30 s, written into the prompt exactly the way the
  model was trained to read it (``duration: <X>s``) -- or left off
* flow-matching sampling: ``num_inference_steps``, ``cfg_scale``,
  ``sigma_shift``, and a ``seed`` for reproducible takes
* an optional negative prompt, steered by the CFG weight
* batch generation: queue several captions and get one take per caption
* a clearly-labelled offline demo synth so the whole interface stays
  explorable without a GPU

The pipeline is a diffusion model, so generation is one buffered request per
take: nothing to stream, but every solver step is reported through
``GET /api/generate/progress`` while it runs.

Locally the app runs on top of the sibling ``MOSS-TTS`` checkout with its venv
layered under this one (see ``bootstrap_model_env``). In Docker the venv is a
single environment that contains both the model and the studio, and
``MOSS_CODE_DIR``/``SOUNDEFFECT_MODEL_DIR`` are set by the entrypoint.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

# Every finished take is written here, so the shelf in the UI survives a reload
# and a restart. Override with SOUNDEFFECT_OUTPUTS_DIR (the image points it at a
# directory that can be mounted as a volume).
#
# `or` rather than a get() default, here and below: a variable that is SET BUT
# EMPTY would otherwise win, and Path("") is the current directory.
OUTPUTS_DIR = Path(os.environ.get("SOUNDEFFECT_OUTPUTS_DIR") or (APP_DIR / "outputs"))
# The name arrives from the browser, so it is matched against a strict pattern
# (no separators, no dots beyond the extension) before it ever touches the
# filesystem -- and _output_path re-checks containment afterwards.
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.wav$")

# ---------------------------------------------------------------------------
# Model environment
# ---------------------------------------------------------------------------

# Local, layered layout: the model code is the sibling MOSS-TTS checkout and its
# venv (python 3.12, torch, the pipeline deps) is layered UNDER this app's venv,
# so this venv keeps only the four web dependencies (see requirements.txt).
MOSS_CODE_DIR = Path(os.environ.get("MOSS_CODE_DIR") or (APP_DIR.parent / "MOSS-TTS"))
MOSS_VENV_DIR = Path(
    os.environ.get("MOSS_VENV_DIR") or (MOSS_CODE_DIR / "moss_soundeffect_v2" / ".venv")
)

settings = SimpleNamespace(
    # Docker layout: one venv holds everything and the checkout sits at
    # /app/MOSS-SoundEffect-v2.0. A plain repo id or a local dir both work:
    # the pipeline's from_pretrained resolves them. Kept as a STRING, not a
    # Path -- Path() would rewrite "OpenMOSS-Team/MOSS-SoundEffect-v2.0" into
    # a backslash form that huggingface_hub rejects as a repo id.
    model_dir=os.environ.get("SOUNDEFFECT_MODEL_DIR") or "OpenMOSS-Team/MOSS-SoundEffect-v2.0",
    device=os.environ.get("SOUNDEFFECT_DEVICE") or "auto",
    force_demo=False,
    auto_demo=True,
)

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_MAX_SECONDS = 30


def bootstrap_model_env() -> None:
    """Point imports at the model checkout and layer its venv under ours.

    Only the local layered layout does anything here: the model venv's
    site-packages are APPENDED to sys.path, so this app's own venv still wins
    for fastapi/uvicorn/httpx and the model venv supplies torch/numpy/etc.

    In Docker there is one venv and nothing needs layering -- but
    TORCHDYNAMO_DISABLE still has to be decided before torch is first imported,
    because the DiT forward is wrapped in torch.compile and a Triton compile
    error on an unsupported GPU would otherwise kill the first generation.
    """
    if MOSS_CODE_DIR.is_dir():
        code_str = str(MOSS_CODE_DIR)
        if code_str not in sys.path:
            sys.path.insert(0, code_str)
        venv_site = MOSS_VENV_DIR / "Lib" / "site-packages"
        if venv_site.is_dir() and str(venv_site) not in sys.path:
            sys.path.append(str(venv_site))

    # Upstream wraps the DiT with torch.compile + Triton CUDA Graphs; the
    # bundled shell scripts export TORCHDYNAMO_DISABLE=1 for the same reason.
    # A single-venv Docker image sets the same default (see the Dockerfile),
    # but the layered local layout needs it here, before torch is imported.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def _read_model_index(path: Path) -> dict[str, Any]:
    try:
        index_path = path / "model_index.json"
        if index_path.is_file():
            return json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {}


def _model_facts() -> dict[str, Any]:
    local_dir = Path(settings.model_dir)
    index = _read_model_index(local_dir) if local_dir.is_dir() else {}
    return {
        "name": "MOSS-SoundEffect v2.0",
        "model_dir": str(settings.model_dir),
        "device": settings.device,
        "dit_variant": str(index.get("dit_variant", "1.3B")),
        "sample_rate": int(index.get("sample_rate", DEFAULT_SAMPLE_RATE)),
        "channels": 1,
        "max_seconds": int(index.get("max_inference_seconds", DEFAULT_MAX_SECONDS)),
        "languages": ["en", "zh"],
        "architecture": "DiT + Flow Matching, DAC VAE, Qwen3 text encoder",
        "license": "Apache-2.0",
        "model_repo": "https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0",
        "code_repo": "https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_soundeffect_v2",
    }


MODEL_FACTS: dict[str, Any] = _model_facts()
DEFAULT_SAMPLE_RATE = int(MODEL_FACTS["sample_rate"])  # 48 kHz from model_index
DEFAULT_MAX_SECONDS = int(MODEL_FACTS["max_seconds"])  # 30 s from model_index

# ---------------------------------------------------------------------------
# Catalog: the single source of truth for everything the app can do.
# ---------------------------------------------------------------------------

PARAM_RANGES: dict[str, dict[str, Any]] = {
    "seconds": {"min": 0.5, "max": DEFAULT_MAX_SECONDS, "step": 0.5, "default": 10.0},
    "num_inference_steps": {"min": 10, "max": 150, "step": 1, "default": 100},
    "cfg_scale": {"min": 1.0, "max": 8.0, "step": 0.1, "default": 4.0},
    "sigma_shift": {"min": 0.0, "max": 10.0, "step": 0.1, "default": 5.0},
}

QUICK_PROMPTS: dict[str, list[dict[str, str]]] = {
    "en": [
        {"label": "Keyboard typing", "prompt": "The crisp, rhythmic click-clack of fast typing on a mechanical keyboard."},
        {"label": "Rain on a window", "prompt": "Steady rain pattering against a window pane with distant thunder rolling softly outside."},
        {"label": "Dog barking", "prompt": "A dog barking loudly in a park, with birds chirping faintly in the background."},
        {"label": "City street", "prompt": "A busy city street at noon: cars passing, people chatting and a tram bell ringing."},
        {"label": "Ocean waves", "prompt": "Gentle ocean waves rolling onto a sandy beach, seagulls calling overhead."},
        {"label": "Coffee shop", "prompt": "Inside a cozy coffee shop: an espresso machine hissing, cups clinking and quiet chatter."},
        {"label": "Campfire", "prompt": "A crackling campfire in a quiet forest at night, wood popping softly."},
        {"label": "Doorbell", "prompt": "A clear two-tone doorbell chime ringing inside a house."},
        {"label": "Sword clash", "prompt": "Two swords clashing in a fast duel, metal ringing with each impact."},
        {"label": "Footsteps", "prompt": "Slow heavy footsteps walking on a wooden floor in an empty hallway."},
        {"label": "Pouring water", "prompt": "Pouring water into a glass, clear liquid flowing sound, pitch rising as the glass fills up, refreshing."},
    ],
    "zh": [
        {"label": "键盘打字", "prompt": "机械键盘上快速打字的清脆而有节奏的咔嗒声。"},
        {"label": "雨打窗", "prompt": "雨点不断敲打窗户，远处雷声隆隆。"},
        {"label": "狗叫", "prompt": "一只狗在公园里大声地叫，背景中隐约有鸟鸣。"},
        {"label": "城市街道", "prompt": "中午繁忙的城市街道：汽车驶过，人们交谈，电车铃声响起。"},
        {"label": "海浪", "prompt": "温柔的海浪拍打着沙滩，海鸥在头顶鸣叫。"},
        {"label": "咖啡馆", "prompt": "温馨的咖啡馆里：意式咖啡机嘶嘶作响，杯子叮当作响，低声交谈。"},
        {"label": "篝火", "prompt": "夜晚安静的森林里，篝火噼啪作响，木柴轻轻爆裂。"},
        {"label": "倒水", "prompt": "把水倒进玻璃杯，清澈的液体流动声，随着杯子装满音调升高，清爽。"},
        {"label": "鸟鸣", "prompt": "清晨小鸟叽叽喳喳地叫着，叫声清脆悦耳。"},
        {"label": "刷牙", "prompt": "刷牙的声音，牙刷毛摩擦牙齿的那种沙沙声。"},
    ],
}

SOUND_PRESETS: dict[str, list[dict[str, str]]] = {
    "en": [
        {"label": "Nature ambience", "prompt": "A peaceful forest ambience: birds singing in the canopy, a light breeze in the leaves and a distant stream."},
        {"label": "Urban ambience", "prompt": "A crowded intersection in the evening: engines idling, crosswalk beeps and fragments of conversation."},
        {"label": "Creature roar", "prompt": "A huge dragon roaring in a cavernous hall, the roar echoing and rumbling through stone."},
        {"label": "Human action", "prompt": "A person sprinting up metal stairs, footsteps clanging and breathing hard."},
        {"label": "Sci-fi UI", "prompt": "A short, clean sci-fi interface beep with a soft digital shimmer, like a spaceship console confirming an input."},
        {"label": "Whoosh", "prompt": "A fast cinematic whoosh as something large sweeps past the microphone."},
        {"label": "Impact hit", "prompt": "A deep cinematic impact hit with a sub-bass boom and a metallic tail."},
        {"label": "Percussive loop", "prompt": "A short rhythmic percussion loop: hands clapping and fingers snapping in a steady groove."},
    ],
    "zh": [
        {"label": "自然环境", "prompt": "宁静的森林环境：树冠上鸟儿歌唱，微风吹过树叶，远处有小溪。"},
        {"label": "城市环境", "prompt": "傍晚拥挤的十字路口：发动机怠速，人行横道的提示音，零星的谈话声。"},
        {"label": "生物吼叫", "prompt": "一条巨龙在洞穴大厅中咆哮，吼声在石头间回荡轰鸣。"},
        {"label": "人物动作", "prompt": "一个人快步跑上金属楼梯，脚步哐当作响，呼吸急促。"},
        {"label": "科幻提示音", "prompt": "一声干净短促的科幻界面提示音，带着柔和的数字闪光，像飞船控制台确认输入。"},
        {"label": "快速呼啸", "prompt": "快速的电影式呼啸声，像某个巨大的物体从麦克风旁掠过。"},
        {"label": "重击", "prompt": "深沉的电影式重击声，带次低音轰鸣和金属尾音。"},
    ],
}

NEGATIVE_PRESETS: list[dict[str, str]] = [
    {"label": "No speech", "prompt": "speech, talking, voices, narration"},
    {"label": "No music", "prompt": "music, singing, melody"},
    {"label": "Clean", "prompt": "no distortion, no clipping, no background noise"},
    {"label": "No hum", "prompt": "electrical hum, buzz, static"},
]

PARAM_DOCS: list[dict[str, str]] = [
    {"param": "prompt", "detail": "Caption in English or Chinese. With the duration tag on, “ duration: <X>s” is appended exactly as at training time."},
    {"param": "negative_prompt", "detail": "CFG negative prompt. Empty by default; steer it with CFG scale (1.0 disables CFG entirely)."},
    {"param": "seconds", "detail": "Output duration. The pipeline always denoises a fixed 30 s latent and crops the waveform to this length."},
    {"param": "num_inference_steps", "detail": "Flow-match solver steps. More steps: slower, cleaner. 100 is the recommended default."},
    {"param": "cfg_scale", "detail": "Classifier-free guidance weight. 4.0 is recommended; 1.0 turns guidance off and the negative prompt is ignored."},
    {"param": "sigma_shift", "detail": "Flow-match scheduler shift applied per call. 5.0 is the upstream default."},
    {"param": "seed", "detail": "RNG seed for the noise initializer. Same seed + same settings = the same take."},
    {"param": "append_duration_suffix", "detail": "Off writes the caption verbatim; the tag is how the model was trained to read duration."},
    {"param": "batch prompts", "detail": "Queue several captions: one take per caption, all with the same settings."},
]

# ---------------------------------------------------------------------------
# Output shelf
# ---------------------------------------------------------------------------


def _output_path(name: str) -> Path:
    """Resolve a requested output name, refusing anything but a bare .wav.

    The pattern is what rejects ``..\\..\\secrets.wav`` early, and the
    containment check is what still holds if the pattern is ever loosened.
    """
    if not SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Not a valid output name.")
    path = (OUTPUTS_DIR / name).resolve()
    if path.parent != OUTPUTS_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No such output: {name}")
    return path


def _wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """16-bit PCM WAV from raw little-endian s16le samples, built in memory."""
    data_len = len(pcm)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_len,
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,   # PCM
        channels,
        sample_rate,
        sample_rate * channels * 2,  # byte rate
        channels * 2,                # block align
        16,                          # bits per sample
        b"data",
        data_len,
    ) + pcm


def _save_output(
    wav_data: bytes,
    request: dict[str, Any],
    *,
    sample_rate: int,
    channels: int,
    duration: float,
    demo: bool = False,
    total_ms: float | None = None,
    steps_completed: int | None = None,
    batch_index: int | None = None,
) -> dict[str, Any]:
    """Write one take as WAV plus a JSON sidecar, and return its record.

    The sidecar is what makes the shelf survivable: the list endpoint is built
    from these files, so a take outlives the browser tab, the server process and
    a container restart without any database.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"sfx-{stamp}-{request.get('mode', 'take')}-seed{request.get('seed', 0)}"
    name = f"{stem}.wav"
    counter = 2
    while (OUTPUTS_DIR / name).exists():  # two takes in the same second are normal
        name = f"{stem}-{counter}.wav"
        counter += 1

    path = OUTPUTS_DIR / name
    path.write_bytes(wav_data)
    record = {
        **request,
        "name": name,
        "url": f"/api/outputs/{name}",
        "sample_rate": sample_rate,
        "channels": channels,
        "audio_bytes": len(wav_data),
        "duration": duration,
        "demo": demo,
        "total_ms": total_ms,
        "steps_completed": steps_completed,
        "batch_index": batch_index,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_ts": time.time(),
    }
    path.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def _load_outputs(limit: int = 200) -> list[dict[str, Any]]:
    """Every saved take, newest first, skipping sidecars with no audio beside them."""
    if not OUTPUTS_DIR.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for sidecar in OUTPUTS_DIR.glob("*.json"):
        if not sidecar.with_suffix(".wav").is_file():
            continue  # audio deleted by hand: the sidecar is now a ghost
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # Defaults, not just for tidiness: a sidecar written by an older build
        # is still a valid take, and the shelf must not reject it for a field
        # that did not exist when it was produced.
        record.setdefault("name", sidecar.with_suffix(".wav").name)
        record.setdefault("url", f"/api/outputs/{record['name']}")
        record.setdefault("demo", False)
        record.setdefault("duration", 0.0)
        record.setdefault("total_ms", None)
        record.setdefault("prompt", "")
        records.append(record)
    records.sort(key=lambda r: r.get("created_ts", 0), reverse=True)
    return records[:limit]


# ---------------------------------------------------------------------------
# The engine: the real pipeline, loaded at most once
# ---------------------------------------------------------------------------

_ENGINE_LOCK = threading.Lock()
_PIPELINE_LOCK = threading.Lock()  # one generation at a time; one pipeline copy
_ENGINE_STATE: dict[str, Any] = {"status": "idle", "detail": "", "loaded_at": None, "pipe": None}


class _ProgressCapture:
    """Stand-in for tqdm that counts completed diffusion steps.

    The pipeline calls ``progress_bar_cmd(iterable)`` around the timestep loop,
    so the counting wrapper below sees every step; ``total`` is the requested
    ``num_inference_steps``. A display value read across threads needs no lock.
    """

    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.completed = 0
        self.last_t = 0.0
        self._last_per = 0.0
        self.eta_s: float | None = None

    def __call__(self, iterable):  # noqa: ANN001 - mirrors tqdm's signature
        return _StepCounter(self, iterable)

    def step_done(self) -> None:
        self.completed += 1
        now = time.perf_counter()
        if self.last_t:
            per_step = (now - self.last_t) if self.completed == 1 else (
                (now - self.last_t) * 0.2 + self._last_per * 0.8
            )
            self._last_per = per_step
            self.eta_s = max(0.0, per_step * max(0, self.total - self.completed))
        self.last_t = now


class _StepCounter:
    def __init__(self, capture: _ProgressCapture, iterable) -> None:
        self._capture = capture
        self._iter = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._iter)
        self._capture.step_done()
        return item


# The progress endpoint reads the capture that the pipeline is actually
# stepping through; _PIPELINE_LOCK serialises generation, so there is at most
# one running at any moment.
_CURRENT_PROGRESS: _ProgressCapture | None = None
_PROGRESS_LOCK = threading.Lock()


def _engine_status() -> dict[str, Any]:
    state = _ENGINE_STATE["status"]
    return {
        "engine": state,
        "detail": _ENGINE_STATE["detail"],
        "loaded_at": _ENGINE_STATE["loaded_at"],
        "demo": state != "ready",
        "fallback": settings.auto_demo,
        "model_dir": MODEL_FACTS["model_dir"],
        "device": MODEL_FACTS["device"],
        "sample_rate": MODEL_FACTS["sample_rate"],
        "max_seconds": MODEL_FACTS["max_seconds"],
        "upstream": MODEL_FACTS["model_dir"],
        "connected": state == "ready",
    }


def _load_engine() -> Any:
    """Load ``MossSoundEffectPipeline`` once, from the local dir or the Hub.

    Loading is done synchronously under the engine lock: ``from_pretrained``
    pulls ~10 GB of weights on first use, and the generation thread is the one
    that should wait. Two generations racing here would otherwise build two
    copies of a 10 GB model.
    """
    with _ENGINE_LOCK:
        if _ENGINE_STATE["status"] == "ready":
            return _ENGINE_STATE["pipe"]
        if _ENGINE_STATE["status"] == "failed":
            raise HTTPException(
                status_code=503,
                detail=(
                    "The model failed to load earlier and will not be retried "
                    f"until the server restarts: {_ENGINE_STATE['detail']}"
                ),
            )
        _ENGINE_STATE["status"] = "loading"
        _ENGINE_STATE["detail"] = "importing torch and the pipeline"
        started = time.perf_counter()
        try:
            import torch  # kept lazy: TORCHDYNAMO is decided in bootstrap_model_env

            device = settings.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            print(
                f"[studio] loading MOSS-SoundEffect from {settings.model_dir} "
                f"on {device} (torch {torch.__version__}) ...",
                flush=True,
            )
            _ENGINE_STATE["detail"] = f"loading weights onto {device}"
            from moss_soundeffect_v2 import MossSoundEffectPipeline

            pipe = MossSoundEffectPipeline.from_pretrained(
                str(settings.model_dir),
                torch_dtype=torch.bfloat16,
                device=device,
            )
            _ENGINE_STATE["pipe"] = pipe
            _ENGINE_STATE["status"] = "ready"
            _ENGINE_STATE["detail"] = f"ready on {device}"
            _ENGINE_STATE["loaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(f"[studio] pipeline ready in {time.perf_counter() - started:.1f}s", flush=True)
            return pipe
        except Exception as exc:  # noqa: BLE001 - the load path must never crash the app
            _ENGINE_STATE["status"] = "failed"
            _ENGINE_STATE["detail"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load the MOSS pipeline: {type(exc).__name__}: {exc}",
            ) from exc


def _engine_generate(
    *,
    prompts: list[str],
    seconds: float,
    steps: int,
    cfg_scale: float,
    sigma_shift: float,
    seed: int,
    negative_prompt: str,
    append_duration_suffix: bool,
    progress: _ProgressCapture,
) -> tuple[list[bytes], int, int, float]:
    """Run one pipeline call; return (wavs as s16le mono, sample_rate, channels, duration)."""
    import numpy as np
    import torch

    pipe = _load_engine()
    with _PIPELINE_LOCK:
        audio = pipe(
            prompt=prompts,
            seconds=seconds,
            num_inference_steps=steps,
            cfg_scale=cfg_scale,
            sigma_shift=sigma_shift,
            seed=seed,
            negative_prompt=negative_prompt,
            append_duration_suffix=append_duration_suffix,
            progress_bar_cmd=progress,
        )

    sample_rate = int(pipe.sample_rate)
    audio_np = audio.detach().float().cpu().numpy()
    if audio_np.ndim == 2:  # a single take arrives as (C, T)
        audio_np = audio_np[np.newaxis, ...]
    wavs: list[bytes] = []
    for take in audio_np:
        mono = take[0] if take.shape[0] == 1 else np.transpose(take)
        pcm = np.clip(np.nan_to_num(mono.astype(np.float32), nan=0.0), -1.0, 1.0)
        wavs.append((pcm * 32767.0).astype("<i2").tobytes())
    channels = 1
    duration = len(wavs[0]) / 2 / sample_rate if wavs else 0.0
    return wavs, sample_rate, channels, duration


# ---------------------------------------------------------------------------
# Demo fallback
# ---------------------------------------------------------------------------


def _demo_pcm(prompt: str, seconds: float) -> bytes:
    """A small formant-ish placeholder synth so the UI works without a GPU.

    It is intentionally obvious that this is *not* the model: it is a buzzy
    robotic soundscape whose character tracks the words in the caption.
    """
    import array as _array

    sample_rate = DEFAULT_SAMPLE_RATE
    total = max(0.5, min(seconds, DEFAULT_MAX_SECONDS))
    count = int(sample_rate * total)
    seed = sum(ord(c) for c in prompt) or 7
    samples = _array.array("h", bytes(count * 2))
    phase = 0.0
    for i in range(count):
        t = i / sample_rate
        # A slowly wandering pair of siren-ish tones over a low drone.
        wander = math.sin(2 * math.pi * (0.11 + (seed % 13) * 0.004) * t)
        freq = 96.0 * (1.0 + 0.35 * wander) + (seed % 7) * 3.0
        phase += 2 * math.pi * freq / sample_rate
        value = (
            math.sin(phase) * 0.42
            + math.sin(phase * 2.01 + wander) * 0.18
            + math.sin(phase * 0.5) * 0.2
        )
        # Amplitude swells so it reads as an ambience rather than a beep.
        env = 0.55 + 0.45 * math.sin(2 * math.pi * 0.23 * t + seed)
        fade = min(1.0, t / 0.05, max(0.0, (total - t) / 0.05))
        samples[i] = int(max(-1.0, min(1.0, value * env * fade * 0.5)) * 32767)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def _demo_wav(request_record: dict[str, Any], seconds: float, reason: str) -> Response:
    pcm = _demo_pcm(request_record.get("prompt", ""), seconds)
    total_ms = round(0.0, 1)
    record = _save_output(
        _wav_bytes(pcm, DEFAULT_SAMPLE_RATE, 1),
        request_record,
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=1,
        duration=len(pcm) / 2 / DEFAULT_SAMPLE_RATE,
        demo=True,
        total_ms=total_ms,
    )
    return Response(
        content=_wav_bytes(pcm, DEFAULT_SAMPLE_RATE, 1),
        media_type="audio/wav",
        headers={
            "X-SoundEffect-Demo": "1",
            "X-SoundEffect-Warning": f"demo audio ({reason})".encode("ascii", "replace").decode("ascii"),
            "X-SoundEffect-Mode": request_record["mode"],
            "X-SoundEffect-Output": record["name"],
            "X-SoundEffect-Sample-Rate": str(DEFAULT_SAMPLE_RATE),
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _parse_params(
    seconds: float, steps: int, cfg_scale: float, sigma_shift: float, seed: int
) -> tuple[float, int, float, float, int]:
    r = PARAM_RANGES
    if not math.isfinite(seconds) or seconds <= 0:
        raise HTTPException(status_code=400, detail="seconds must be greater than 0.")
    if seconds > r["seconds"]["max"] + 1e-6:
        raise HTTPException(
            status_code=400,
            detail=f"seconds exceeds the model's {r['seconds']['max']:.0f}s limit.",
        )
    if not 1 <= steps <= 500:
        raise HTTPException(status_code=400, detail="steps must be between 1 and 500.")
    if not math.isfinite(cfg_scale) or cfg_scale <= 0:
        raise HTTPException(status_code=400, detail="cfg_scale must be greater than 0.")
    if not math.isfinite(sigma_shift) or sigma_shift < 0:
        raise HTTPException(status_code=400, detail="sigma_shift must be >= 0.")
    if seed < 0:
        raise HTTPException(status_code=400, detail="seed must be >= 0.")
    return round(float(seconds), 1), int(steps), float(cfg_scale), float(sigma_shift), int(seed)


async def _run_generation(
    *,
    prompts: list[str],
    seconds: float,
    steps: int,
    cfg_scale: float,
    sigma_shift: float,
    seed: int,
    negative_prompt: str,
    append_duration_suffix: bool,
    save: bool = True,
) -> Response:
    seconds, steps, cfg_scale, sigma_shift, seed = _parse_params(
        seconds, steps, cfg_scale, sigma_shift, seed
    )
    prompts = [p.strip() for p in prompts if p.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    prompts = prompts[:10]

    request_record = {
        "mode": "batch" if len(prompts) > 1 else "sfx",
        "prompts": prompts,
        "prompt": prompts[0],
        "seconds": seconds,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sigma_shift": sigma_shift,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "duration_tag": append_duration_suffix,
    }

    # --demo short-circuits before the engine is ever touched: it exists so the
    # whole interface is explorable on a machine without the model.
    if settings.force_demo:
        return _demo_wav(request_record, seconds, "demo mode was requested with --demo")

    progress = _ProgressCapture(steps)
    global _CURRENT_PROGRESS
    with _PROGRESS_LOCK:
        _CURRENT_PROGRESS = progress
    started = time.perf_counter()

    def work() -> tuple[list[bytes], int, int, float]:
        try:
            return _engine_generate(
                prompts=prompts,
                seconds=seconds,
                steps=steps,
                cfg_scale=cfg_scale,
                sigma_shift=sigma_shift,
                seed=seed,
                negative_prompt=negative_prompt,
                append_duration_suffix=append_duration_suffix,
                progress=progress,
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {type(exc).__name__}: {exc}",
            ) from exc

    try:
        wavs, sample_rate, channels, duration = await run_in_threadpool(work)
    except HTTPException as exc:
        if exc.status_code == 503 and settings.auto_demo and not settings.force_demo:
            return _demo_wav(request_record, seconds, "model not loaded")
        raise
    finally:
        with _PROGRESS_LOCK:
            if _CURRENT_PROGRESS is progress:
                _CURRENT_PROGRESS = None

    total_ms = round((time.perf_counter() - started) * 1000, 1)
    saved_names: list[str] = []
    if save:
        # Each take is written as its own WAV + sidecar; a batch shares the
        # settings and differs by prompt (recorded per sidecar).
        for index, pcm in enumerate(wavs):
            record = _save_output(
                _wav_bytes(pcm, sample_rate, channels),
                {
                    **request_record,
                    "prompt": prompts[index],
                    "mode": "batch" if len(prompts) > 1 else "sfx",
                },
                sample_rate=sample_rate,
                channels=channels,
                duration=len(pcm) / 2 / sample_rate,
                demo=False,
                total_ms=total_ms,
                steps_completed=progress.completed,
                batch_index=index if len(wavs) > 1 else None,
            )
            saved_names.append(record["name"])

    if len(wavs) == 1:
        headers = {
            "X-SoundEffect-Mode": request_record["mode"],
            "X-SoundEffect-Elapsed-Ms": str(total_ms),
            "X-SoundEffect-Sample-Rate": str(sample_rate),
            "X-SoundEffect-Steps-Done": str(progress.completed),
            "Cache-Control": "no-store",
        }
        if saved_names:
            headers["X-SoundEffect-Output"] = saved_names[0]
        return Response(
            content=_wav_bytes(wavs[0], sample_rate, channels),
            media_type="audio/wav",
            headers=headers,
        )

    # Batch: one JSON document of WAV payloads (base64) -- the UI splits them.
    return JSONResponse(
        {
            "count": len(wavs),
            "sample_rate": sample_rate,
            "saved": saved_names,
            "elapsed_ms": total_ms,
            "wavs_b64": [base64.b64encode(_wav_bytes(p, sample_rate, channels)).decode("ascii") for p in wavs],
        },
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(_: FastAPI):
    yield


app = FastAPI(title="TostAI Sound Effect Studio", version="1.0.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/catalog")
async def catalog() -> JSONResponse:
    return JSONResponse(
        {
            "param_ranges": PARAM_RANGES,
            "param_docs": PARAM_DOCS,
            "quick_prompts": QUICK_PROMPTS,
            "sound_presets": SOUND_PRESETS,
            "negative_presets": NEGATIVE_PRESETS,
            "model": MODEL_FACTS,
        }
    )


@app.get("/api/status")
async def status() -> JSONResponse:
    if settings.force_demo:
        payload = _engine_status()
        payload.update({"connected": False, "detail": "Forced demo mode (--demo)."})
        return JSONResponse(payload)
    return JSONResponse(_engine_status())


@app.post("/api/generate")
async def generate(req: Request) -> Response:
    """Buffered synthesis returned as a WAV file (single prompt) or JSON batch."""
    form = await req.form()
    try:
        seconds = float(form.get("seconds") or PARAM_RANGES["seconds"]["default"])
        steps = int(form.get("steps") or PARAM_RANGES["num_inference_steps"]["default"])
        cfg_scale = float(form.get("cfg_scale") or PARAM_RANGES["cfg_scale"]["default"])
        sigma_shift = float(form.get("sigma_shift") or PARAM_RANGES["sigma_shift"]["default"])
        seed = int(form.get("seed") or 0)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Numeric parameter out of range.") from None

    prompt_value = str(form.get("prompts") or form.get("prompt") or "")
    return await _run_generation(
        prompts=prompt_value.splitlines(),
        seconds=seconds,
        steps=steps,
        cfg_scale=cfg_scale,
        sigma_shift=sigma_shift,
        seed=seed,
        negative_prompt=str(form.get("negative_prompt") or ""),
        append_duration_suffix=str(form.get("duration_tag") or "1") not in ("0", "false"),
        save=str(form.get("save") or "1") not in ("0", "false"),
    )


@app.get("/api/generate/progress")
async def generate_progress() -> JSONResponse:
    progress = _CURRENT_PROGRESS
    if progress is None:
        return JSONResponse({"active": False, "steps_completed": 0, "steps_total": 0})
    return JSONResponse(
        {
            "active": True,
            "steps_completed": progress.completed,
            "steps_total": progress.total,
            "eta_s": round(progress.eta_s, 1) if progress.eta_s is not None else None,
        }
    )


@app.get("/api/outputs")
async def list_outputs() -> JSONResponse:
    """Every saved take, newest first -- this is what the UI's shelf renders."""
    outputs = _load_outputs()
    return JSONResponse({"dir": str(OUTPUTS_DIR), "count": len(outputs), "outputs": outputs})


@app.get("/api/outputs/{name}")
async def get_output(name: str) -> FileResponse:
    return FileResponse(
        _output_path(name), media_type="audio/wav", headers={"Cache-Control": "no-store"}
    )


@app.delete("/api/outputs/{name}")
async def delete_output(name: str) -> JSONResponse:
    path = _output_path(name)
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return JSONResponse({"deleted": name})


# --------------------------------------------------------------------------- #
# self-update: pull the latest source from this app's own repo, then restart
# --------------------------------------------------------------------------- #
#
# The source repository is public, so the update needs no credential: the
# commits URL and the tarball URL below both answer without authorization. A
# per-request "token" field is still accepted and, when supplied, sent as a
# Bearer header -- it only lifts the anonymous GitHub rate limit on a busy
# network. Nothing is ever written to disk.
#
# Why the restart is a re-exec on POSIX: index.html/app.js are re-read per
# request, but server.py is Python held in memory, so new code on disk is
# invisible until the process restarts. os.execv replaces the process image in
# place, which keeps the same PID 1 and the same container. On Windows execv is
# not usable from a thread (measured in the reference studio), so a detached
# replacement is spawned instead.

APP_REPO = os.environ.get("TOSTAI_APP_REPO", "camenduru/TostAI-Sound-Effect-Studio")
APP_REV_FILE = APP_DIR / ".tostai_rev"
UPDATE_BACKUP = APP_DIR / ".update_backup"

# Never overwritten by an update. `outputs/` is this container's own state and
# is gitignored upstream. `.update_backup` is excluded so a backup never
# contains itself; `.venv` because the environment is not source.
UPDATE_KEEP = ("outputs", ".update_backup", ".git", "__pycache__", ".venv", ".freebuff")

_UA = {
    "User-Agent": "tostai-sound-effect-studio-update",
    "Accept": "application/vnd.github+json",
}


def _gh_json(url: str, token: str | None = None) -> tuple[Any, str | None]:
    """GET a GitHub API URL. Returns (data, None) or (None, human_error)."""
    headers = dict(_UA)
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "GitHub rejected the token (401). It may have expired."
        if e.code == 404:
            return None, (
                f"GitHub returned 404 for {url}. Either the repository name is "
                "wrong, or the repository is unavailable."
            )
        if e.code == 403:
            return None, "GitHub returned 403 (rate limit or blocked). Try again later."
        return None, f"GitHub returned HTTP {e.code}."
    except Exception as ex:  # noqa: BLE001
        return None, f"{type(ex).__name__}: {ex}"


def _current_rev() -> str:
    try:
        return APP_REV_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _restart_soon(delay: float = 1.5) -> None:
    """Replace this process with a fresh one, once the response has flushed.

    The delay is load-bearing: both paths below tear the server down, so doing
    it before the response is written turns a successful update into a network
    error on the user's screen.
    """

    def _go() -> None:
        time.sleep(delay)
        argv = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
        try:
            if os.name == "nt":
                # Windows: os.execv is NOT usable from here -- a uvicorn server
                # that execv's itself from a background thread dies outright:
                # the in-flight response is cut off mid-body, no replacement
                # process is ever started, and the port goes dead. So spawn a
                # detached replacement and leave; exiting is safe because there
                # is no PID 1 to keep alive.
                flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                subprocess.Popen(argv, cwd=str(APP_DIR), close_fds=True, creationflags=flags)
                os._exit(0)
            else:
                # POSIX, and specifically a container: execve(2) swaps the image
                # inside the SAME pid, so PID 1 stays PID 1 and the container is
                # not stopped. Exiting would end the container, and the default
                # restart policy is `no`.
                os.execv(sys.executable, argv)
        except Exception:  # noqa: BLE001
            # Restart failed, so this process is still serving the OLD code. Say
            # so rather than exiting: a stopped container is worse than a stale
            # one that still works.
            traceback.print_exc()

    threading.Thread(target=_go, daemon=True, name="tostai-restart").start()


# The commit subject the running process started from. Set on a successful POST
# and read by the GET below, so the dialog can name what is installed rather
# than only its hash.
_rev_subject = ""


@app.get("/api/update")
async def update_status() -> JSONResponse:
    """What is installed, so the dialog can say so before anything is typed."""
    rev = _current_rev()
    return JSONResponse(
        {"ok": True, "repo": APP_REPO, "rev": rev, "short": rev[:10], "subject": _rev_subject}
    )


@app.post("/api/update")
async def update_apply(req: Request) -> JSONResponse:
    global _rev_subject
    try:
        body = await req.json()
    except Exception:  # noqa: BLE001
        body = {}
    token = ((body or {}).get("token") or "").strip() or None

    # 1. What is upstream right now?
    commits, err = _gh_json(f"https://api.github.com/repos/{APP_REPO}/commits?per_page=1", token)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    if not commits:
        return JSONResponse({"ok": False, "error": "the repository has no commits"}, status_code=400)
    latest = commits[0]["sha"]
    subject = (commits[0].get("commit", {}).get("message") or "").splitlines()[0]
    current = _current_rev()

    if latest == current:
        _rev_subject = subject
        return JSONResponse(
            {
                "ok": True,
                "updated": False,
                "rev": latest,
                "subject": subject,
                "message": f"already up to date at {latest[:10]}",
            }
        )

    tmp = tempfile.mkdtemp(prefix="tostai_update_")
    try:
        # 2. Fetch that exact tree. The repo is public, so the tarball URL
        #    needs no Authorization header -- and none is sent, so a caller
        #    supplied token (rate-limit relief for the API call above) can
        #    never leak to the redirect host.
        tarball = os.path.join(tmp, "src.tar.gz")
        dl = urllib.request.Request(
            f"https://api.github.com/repos/{APP_REPO}/tarball/{latest}", headers=dict(_UA)
        )
        try:
            with urllib.request.urlopen(dl, timeout=180) as r, open(tarball, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as ex:  # noqa: BLE001
            return JSONResponse(
                {"ok": False, "error": f"download failed: {type(ex).__name__}: {ex}"},
                status_code=400,
            )

        root = os.path.join(tmp, "x")
        os.makedirs(root)
        with tarfile.open(tarball, "r:gz") as tf:
            try:
                tf.extractall(root, filter="data")  # py3.12+
            except TypeError:
                tf.extractall(root)  # py3.10/3.11

        # GitHub wraps the tree in a single <owner>-<repo>-<shortsha> directory.
        entries = sorted(e for e in os.listdir(root) if not e.startswith("."))
        if len(entries) != 1 or not os.path.isdir(os.path.join(root, entries[0])):
            return JSONResponse(
                {"ok": False, "error": f"unexpected tarball layout: {entries!r}"}, status_code=400
            )
        src = os.path.join(root, entries[0])

        if not os.path.isfile(os.path.join(src, "server.py")):
            return JSONResponse(
                {"ok": False, "error": "refusing to install: the new tree has no server.py"},
                status_code=400,
            )

        # 3. Parse every new .py BEFORE anything is swapped in. A file that
        #    fails to parse would be re-exec'd into a container that never
        #    comes back, and the default restart policy is `no`.
        bad = []
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in UPDATE_KEEP]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "rb") as fh:
                        compile(fh.read(), p, "exec")
                except SyntaxError as ex:
                    bad.append(f"{os.path.relpath(p, src)}: line {ex.lineno}: {ex.msg}")
                except (OSError, ValueError) as ex:
                    bad.append(f"{os.path.relpath(p, src)}: {ex}")
        if bad:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "refusing to install -- the new source does not compile:\n"
                    + "\n".join(bad[:10]),
                },
                status_code=400,
            )

        # 4. Back up the current tree, then copy the new one over it. Copy, not
        #    replace, so files deleted upstream linger harmlessly rather than
        #    `outputs/` being swept away with them.
        if UPDATE_BACKUP.is_dir():
            shutil.rmtree(UPDATE_BACKUP, ignore_errors=True)
        try:
            shutil.copytree(APP_DIR, UPDATE_BACKUP, ignore=shutil.ignore_patterns(*UPDATE_KEEP))
        except Exception as ex:  # noqa: BLE001
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"could not write a backup, so nothing was changed: {ex}",
                },
                status_code=400,
            )

        copied = 0
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d not in UPDATE_KEEP]
            rel = os.path.relpath(dirpath, src)
            dst_dir = APP_DIR if rel == "." else APP_DIR / rel
            os.makedirs(dst_dir, exist_ok=True)
            for fn in filenames:
                shutil.copy2(os.path.join(dirpath, fn), os.path.join(dst_dir, fn))
                copied += 1

        APP_REV_FILE.write_text(latest + "\n", encoding="utf-8")
        _rev_subject = subject

        _restart_soon()
        return JSONResponse(
            {
                "ok": True,
                "updated": True,
                "rev": latest,
                "subject": subject,
                "files": copied,
                "message": f"updated to {latest[:10]}, restarting",
            }
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve TostAI Sound Effect Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        default=None,
        help="Model dir or HF repo id (default: $SOUNDEFFECT_MODEL_DIR or the public repo)",
    )
    parser.add_argument(
        "--device", default=None, help="cuda / cpu / auto (default: $SOUNDEFFECT_DEVICE or auto)"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Never load the model; always return demo audio."
    )
    parser.add_argument(
        "--no-demo-fallback",
        action="store_true",
        help="Return a 503 instead of demo audio when the model is unavailable.",
    )
    args = parser.parse_args()

    if args.model:
        settings.model_dir = args.model
        refreshed = _model_facts()
        MODEL_FACTS.clear()
        MODEL_FACTS.update(refreshed)
        globals()["DEFAULT_SAMPLE_RATE"] = int(MODEL_FACTS["sample_rate"])
        globals()["DEFAULT_MAX_SECONDS"] = int(MODEL_FACTS["max_seconds"])
    if args.device:
        settings.device = args.device
        MODEL_FACTS["device"] = args.device
    settings.force_demo = args.demo
    settings.auto_demo = not args.no_demo_fallback

    bootstrap_model_env()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

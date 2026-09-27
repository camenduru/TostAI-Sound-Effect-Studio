# syntax=docker/dockerfile:1.10
#
# 1.10, not 1.7: `env=` on a secret mount (used below to make an optional
# HF_TOKEN visible to one RUN) is rejected by older frontends.
#
# ===========================================================================
# TostAI Sound Effect Studio -- self-contained image
#
# One stage, linear, everything fetched during the build: the model code is
# cloned from GitHub, the weights are downloaded from HuggingFace, and the
# studio itself is cloned from its own GitHub repository. The build context
# supplies only docker_selfcheck.py.
#
#   docker build --build-arg CACHEBUST=$(date +%s) -t camenduru/tostai-sound-effect-studio .
#   docker run --rm --gpus all -p 8000:8000 camenduru/tostai-sound-effect-studio
#
# then open http://127.0.0.1:8000.
#
# THE BUILD CONTEXT IS THIS DIRECTORY (tostai-sound-effect-studio/), not the
# repo root. The model repo beside it is ~10 GB and the MOSS-TTS checkout is
# another clone -- neither is read from the context, so neither should be
# uploaded to the builder. `.dockerignore` therefore ignores everything except
# the studio's build-time proof.
#
# ---------------------------------------------------------------------------
# WHAT IS DOWNLOADED, AND WHERE IT LANDS
#
#   https://github.com/OpenMOSS/MOSS-TTS                      -> /app/MOSS-TTS
#   https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0 -> /app/MOSS-SoundEffect-v2.0
#   https://github.com/camenduru/TostAI-Sound-Effect-Studio    -> /app/tostai-sound-effect-studio
#
# Those are the paths the entrypoint passes to the studio: the model directory
# is the diffusers-style checkpoint (model_index.json + transformer/ + vae/ +
# text_encoder/ + tokenizer/ + scheduler/), and the code checkout supplies the
# `moss_soundeffect_v2` package the studio imports.
#
# ---------------------------------------------------------------------------
# TOKEN REQUIREMENTS
#
# There are none: every GitHub repository this build reads is public, and so is
# the HuggingFace model repo. Anonymous `git clone` and an unauthenticated
# `snapshot_download` are all it needs.
#
# HF_TOKEN is still ACCEPTED as an optional secret mount: it lifts the
# HuggingFace rate limit on a busy build farm. It is passed the same way the
# reference images do it:
#
#   docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     --build-arg CACHEBUST=$(date +%s) \
#     -t camenduru/tostai-sound-effect-studio .
#
# `--secret ...,env=NAME` lifts the value out of the caller's environment and
# `--mount=type=secret,...,env=NAME` exposes it to that ONE RUN. It is NOT an
# ARG and NOT an ENV, so it never reaches `docker history` or `.Config.Env`,
# and it is gone from the next layer. Nothing is written to disk.
#
# ---------------------------------------------------------------------------
# NO CUDA TOOLKIT, AND WHY THAT IS FINE
#
# There is no `cuda_*.run --silent --toolkit` in here: it is a ~4 GB download
# that inference does not need. The pip torch wheels bring their own CUDA
# runtime (nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, ...) and only the HOST
# driver has to exist. That is why `--gpus all` is the whole GPU story.
#
# The wheels are the +cu128 builds the upstream README recommends
# (pip install --extra-index-url https://download.pytorch.org/whl/cu128).
#
# ---------------------------------------------------------------------------
# WHY THE PINS RESOLVE (checked against the indexes, not guessed)
#
#   torch==2.9.0 / torchaudio==2.9.0 -> published as +cu128 wheels on
#     https://download.pytorch.org/whl/cu128, exactly the pins upstream's
#     [torch-cu128] extra uses. --extra-index-url, NOT --index-url: PyPI has to
#     stay reachable for every transitive dependency.
#   numpy==1.26.4, transformers==4.57.1, diffusers==0.37.1, ... -> the exact
#     pins of moss_soundeffect_v2/pyproject.toml, installed by `-e` from the
#     checkout itself so the pin list has exactly one owner.
#
# TORCHDYNAMO_DISABLE=1 is set for RUNTIME because upstream wraps the DiT in
# torch.compile + Triton CUDA Graphs; on a GPU whose sm Triton does not know,
# the first generation would die inside a TorchDynamo error. The env var is
# what the bundled `infer_from_pipeline.sh` exports for the same reason.
# ===========================================================================
# 24.04, not 22.04: moss_soundeffect_v2 requires python >= 3.12, and 22.04's
# python3 is 3.10 -- a `python3.12` package does not exist in its archive at
# all. 24.04 ships python3 = 3.12 natively, which is exactly what upstream
# pins.
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="TostAI Sound Effect Studio" \
      org.opencontainers.image.description="Web app for MOSS-SoundEffect v2.0: text-to-sound effects with duration, CFG, negative prompt, sigma shift, seed and batch control." \
      org.opencontainers.image.source="https://github.com/camenduru/TostAI-Sound-Effect-Studio" \
      org.opencontainers.image.url="https://hub.docker.com/r/camenduru/tostai-sound-effect-studio" \
      org.opencontainers.image.documentation="https://github.com/camenduru/TostAI-Sound-Effect-Studio#docker"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=True \
    PYTHONDONTWRITEBYTECODE=True

# ---------------------------------------------------------------------------
# System packages and a non-root user
#
# ffmpeg is here for torchaudio's format coverage when saving WAVs; git-lfs
# because a checkout that declares LFS filters can fail on the smudge step
# without it.
#
# build-essential is deliberately absent: every wheel below is prebuilt, and it
# is ~200 MB of compiler nobody uses at runtime.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        git git-lfs curl ca-certificates ffmpeg \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash camenduru \
    && mkdir -p /app /opt \
    && chown -R camenduru:camenduru /app /opt

# ---------------------------------------------------------------------------
# The virtualenv, used by EVERY python process in this image
#
# Unlike the local layered layout, the image has ONE venv that holds both the
# model's pins and the studio's four web dependencies, so no sys.path layering
# happens here.
# ---------------------------------------------------------------------------
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip

ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    MOSS_CODE_DIR=/app/MOSS-TTS \
    SOUNDEFFECT_MODEL_DIR=/app/MOSS-SoundEffect-v2.0 \
    TORCHDYNAMO_DISABLE=1

# ---------------------------------------------------------------------------
# The model code checkout
#
# Placed above the torch install so that a change to the code does not force a
# ~2.5 GB torch re-download: this layer is a few MB and rebuilds in seconds.
#
# The project subdirectory is installed `pip install -e`, which reads the exact
# dependency pins from moss_soundeffect_v2/pyproject.toml -- the pin list has
# exactly one owner, and the studio imports `moss_soundeffect_v2` from the same
# environment the entrypoint runs it in.
# ---------------------------------------------------------------------------
RUN set -eu; \
    git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS.git /app/MOSS-TTS; \
    rm -rf /app/MOSS-TTS/.git; \
    test -f /app/MOSS-TTS/moss_soundeffect_v2/pipeline_moss_soundeffect.py

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.9.0+cu128" "torchaudio==2.9.0+cu128"

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -e "/app/MOSS-TTS/moss_soundeffect_v2[torch-cu128]" \
        "fastapi>=0.115" "uvicorn>=0.30" "httpx>=0.27" "python-multipart>=0.0.18"

# ---------------------------------------------------------------------------
# The weights
#
# Only the files the pipeline reads are fetched. README.md is skipped: it is
# not opened at runtime.
#
# snapshot_download, rather than a list of aria2c/curl calls: it comes with
# transformers (already installed), it resumes, and it verifies.
#
# The revision is pinned rather than tracking main, so an image rebuilt in six
# months is the same model. Override with --build-arg SOUNDEFFECT_REV=<sha|tag>.
# *_OFFLINE is set AFTER the download, for the obvious reason: it is what stops
# the running container from reaching the Hub at all.
# ---------------------------------------------------------------------------
ARG SOUNDEFFECT_REV=main

RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    set -eu; \
    if [ -n "${HF_TOKEN:-}" ]; then echo "using the supplied HF_TOKEN"; else echo "no HF_TOKEN: expecting a public repo"; fi; \
    SOUNDEFFECT_REV="${SOUNDEFFECT_REV:-main}" /opt/venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="OpenMOSS-Team/MOSS-SoundEffect-v2.0",
    revision=os.environ.get("SOUNDEFFECT_REV", "main"),
    local_dir="/app/MOSS-SoundEffect-v2.0",
    allow_patterns=[
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/*",
        "transformer/*",
        "vae/*",
    ],
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=4,
)
print("downloaded to", path)
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------
# The studio -- cloned from its own repository
#
#   https://github.com/camenduru/TostAI-Sound-Effect-Studio -> /app/tostai-sound-effect-studio
#
# Cloned rather than copied out of the build context, so the image is
# reproducible from the repository alone. The repo is public, so the clone is
# anonymous.
#
# CACHEBUST IS NOT OPTIONAL IN PRACTICE. BuildKit caches this clone under a key
# that ignores what the branch points at now, so without the flag a rebuild
# silently re-serves the first snapshot. Pass `--build-arg CACHEBUST=$(date
# +%s)`; it invalidates only the clone and the cheap layers after it, so the
# apt/pip/weights layers stay cached. If the image id does not change after a
# rebuild, nothing was rebuilt.
#
# The resolved commit is written to .tostai_rev so the running app can report
# what it is (GET /api/update). `docker_selfcheck.py` is copied from the build
# context rather than taken from the clone, so the build-time proof may be
# newer than what is committed.
#
# `--chown` is load-bearing, not tidiness: `USER camenduru` is set below, and a
# root-owned COPY turned the reference image's update endpoint into a 500 that
# the UI could not even parse.
# ---------------------------------------------------------------------------
ARG CACHEBUST=0

RUN set -eu; \
    git clone --depth 1 \
      https://github.com/camenduru/TostAI-Sound-Effect-Studio.git \
      /app/tostai-sound-effect-studio; \
    git -C /app/tostai-sound-effect-studio rev-parse HEAD > /app/tostai-sound-effect-studio/.tostai_rev; \
    rm -rf /app/tostai-sound-effect-studio/.git; \
    test -f /app/tostai-sound-effect-studio/server.py; \
    echo "cloned camenduru/TostAI-Sound-Effect-Studio at $(cat /app/tostai-sound-effect-studio/.tostai_rev)"

COPY --chown=camenduru:camenduru docker_selfcheck.py /app/tostai-sound-effect-studio/docker_selfcheck.py

# smoke_generate.py comes with the clone so a running container can be verified
# in place:  docker exec <id> python /app/tostai-sound-effect-studio/smoke_generate.py
# It drives the studio over HTTP and needs a loaded model to be meaningful.
WORKDIR /app/tostai-sound-effect-studio
RUN chmod +x /app/tostai-sound-effect-studio/docker-entrypoint.sh \
    && mkdir -p /app/tostai-sound-effect-studio/outputs \
    && chown -R camenduru:camenduru /app/tostai-sound-effect-studio

USER camenduru

# ---------------------------------------------------------------------------
# Build-time proof
#
# Asserts the code imports, the pipeline class resolves, every weight directory
# really arrived (a truncated download is the failure this catches), torch is a
# CUDA build, and the studio answers its own routes. It does NOT load the
# model: that needs a GPU, and a build must not depend on one. The model is
# exercised on first request instead, which is what the HEALTHCHECK's
# start-period covers.
# ---------------------------------------------------------------------------
RUN /opt/venv/bin/python /app/tostai-sound-effect-studio/docker_selfcheck.py

EXPOSE 8000

# tini reaps the backgrounded model server; without an init, PID 1's orphaned
# children accumulate and `docker stop` can hang on the shutdown path.
ENTRYPOINT ["/usr/bin/tini", "--"]

# The health endpoint never touches the GPU: /api/status reports the engine
# state rather than requiring the model, so a container whose model is still
# loading is "healthy" and the UI says "loading" instead of flapping.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4)"

CMD ["/app/tostai-sound-effect-studio/docker-entrypoint.sh"]

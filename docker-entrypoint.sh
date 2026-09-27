#!/usr/bin/env bash
#
# TostAI Sound Effect Studio entrypoint.
#
# One process, one container: the studio IS the model server. It imports
# `moss_soundeffect_v2` from /app/MOSS-TTS, loads the checkpoint from
# /app/MOSS-SoundEffect-v2.0 on first request, and serves the UI on 0.0.0.0.
#
# SOUNDEFFECT_SERVE_MODEL=0 skips the local model entirely, for the case where
# the weights live elsewhere: the studio then falls back to demo audio (or
# fails, with --no-demo-fallback), and SOUNDEFFECT_MODEL_DIR tells it what to
# try to load.
set -euo pipefail

STUDIO_DIR="${SOUNDEFFECT_STUDIO_DIR:-/app/tostai-sound-effect-studio}"
MODEL_DIR="${SOUNDEFFECT_MODEL_DIR:-/app/MOSS-SoundEffect-v2.0}"
CODE_DIR="${MOSS_CODE_DIR:-/app/MOSS-TTS}"
STUDIO_PORT="${SOUNDEFFECT_STUDIO_PORT:-8000}"

if [ "${SOUNDEFFECT_SERVE_MODEL:-1}" = "1" ]; then
  if [ ! -d "${MODEL_DIR}" ]; then
    echo "studio: no model directory at ${MODEL_DIR}" >&2
    exit 1
  fi
  echo "studio: model  -> ${MODEL_DIR}"
  echo "studio: code   -> ${CODE_DIR}"
else
  echo "studio: SOUNDEFFECT_SERVE_MODEL=0, serving the UI with demo fallback"
fi

echo "studio: UI -> 0.0.0.0:${STUDIO_PORT}"
cd "${STUDIO_DIR}"
exec python server.py --host 0.0.0.0 --port "${STUDIO_PORT}"

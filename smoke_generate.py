"""Exercise TostAI Sound Effect Studio against a running app.

Sends one real generation request (a single take and a two-caption batch),
asserting the things that actually distinguish the paths rather than just
"did it return audio": the WAV header, the saved sidecar, the progress
endpoint, and the traversal guard.

Against a live server:

    python smoke_generate.py                      # demo audio or the model
    python smoke_generate.py --expect-model       # fail if demo audio comes back
    python smoke_generate.py --skip-single        # only the batch path

Exits non-zero if anything fails, so it is usable as a gate.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import httpx

SINGLE_TEXT = "A single woodpecker knocking on a dry tree trunk, steady and rhythmic."
BATCH_TEXTS = [
    "Rain tapping on a tin roof, steady and close.",
    "Wind chimes stirred by a rising breeze on a porch.",
]


def wav_header(data: bytes) -> tuple[float, int, int] | None:
    """Duration, channels and sample rate from a RIFF/WAVE blob."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id, size = struct.unpack_from("<4sI", data, offset)
        body = data[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            _, channels, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", body, 0)
            data_off = offset + 8 + size + (size % 2)
            next_id = data[data_off : data_off + 4]
            if next_id != b"data":
                return None
            data_len = struct.unpack_from("<I", data, data_off + 4)[0]
            return data_len / (sample_rate * channels * bits // 8), channels, sample_rate
        offset += 8 + size + (size % 2)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--expect-model", action="store_true",
                        help="fail if the server serves demo audio")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-batch", action="store_true")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    try:
        status = httpx.get(f"{base}/api/status", timeout=10).json()
    except Exception as exc:
        raise SystemExit(f"no app server at {base}: {exc}") from exc
    print(f"app    : {base}")
    print(f"engine : {status.get('engine')} ({status.get('detail')})")
    print(f"model  : {status.get('model_dir')}")
    print()

    failures: list[str] = []
    client = httpx.Client(timeout=3600.0)

    def run(fields: dict, label: str) -> None:
        started = time.perf_counter()
        response = client.post(f"{base}/api/generate", data=fields)
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            failures.append(f"{label}: HTTP {response.status_code}: {response.text[:200]}")
            print(f"FAIL  {label}: HTTP {response.status_code}")
            return
        demo = response.headers.get("X-SoundEffect-Demo") == "1"
        saved = response.headers.get("X-SoundEffect-Output", "-")
        media = response.headers.get("content-type", "")
        print(f"{'DEMO' if demo else 'ok  '}  {label}: {media} {len(response.content)} bytes "
              f"in {elapsed:.1f}s  saved={saved}")
        if args.expect_model and demo:
            failures.append(f"{label}: served demo audio (expected the model)")
        header = wav_header(response.content)
        if header is None:
            failures.append(f"{label}: response is not a well-formed RIFF/WAVE file")
            return
        duration, channels, sample_rate = header
        print(f"      wav: {duration:.2f}s, {channels}ch, {sample_rate} Hz")
        if sample_rate != 48000 or channels != 1:
            failures.append(f"{label}: {channels}ch {sample_rate}Hz, expected 1ch 48000Hz")
        if not 0.4 <= duration <= 31:
            failures.append(f"{label}: implausible duration {duration:.2f}s")
        if saved == "-":
            failures.append(f"{label}: no X-SoundEffect-Output header: the take was not saved")
            return
        # The sidecar must agree with what came down the wire.
        listing = client.get(f"{base}/api/outputs").json()
        record = next((o for o in listing.get("outputs", []) if o["name"] == saved), None)
        if record is None:
            failures.append(f"{label}: {saved} is not in GET /api/outputs")
            return
        if abs((record.get("duration") or 0) - duration) > 0.05:
            failures.append(f"{label}: sidecar duration {record.get('duration')} != WAV {duration:.2f}")
        if record.get("sample_rate") != sample_rate:
            failures.append(f"{label}: sidecar sample_rate mismatch")
        if bool(record.get("demo")) != demo:
            failures.append(f"{label}: sidecar demo flag mismatch")

    if not args.skip_single:
        run(
            {
                "prompts": SINGLE_TEXT,
                "seconds": "2.0",
                "steps": "10",
                "cfg_scale": "4.0",
                "sigma_shift": "5.0",
                "seed": "7",
            },
            "single",
        )

    if not args.skip_batch:
        response = client.post(
            f"{base}/api/generate",
            data={
                "prompts": "\n".join(BATCH_TEXTS),
                "seconds": "2.0",
                "steps": "10",
                "cfg_scale": "4.0",
                "seed": "7",
            },
        )
        if response.status_code != 200:
            failures.append(f"batch: HTTP {response.status_code}: {response.text[:200]}")
            print(f"FAIL  batch: HTTP {response.status_code}")
        else:
            payload = response.json()
            saved = payload.get("saved", [])
            print(f"ok    batch: {payload.get('count')} takes in {payload.get('elapsed_ms')} ms")
            if payload.get("count") != len(BATCH_TEXTS):
                failures.append(f"batch: count {payload.get('count')} != {len(BATCH_TEXTS)}")
            if len(saved) != len(BATCH_TEXTS):
                failures.append(f"batch: {len(saved)} saved names != {len(BATCH_TEXTS)}")
            else:
                listing = client.get(f"{base}/api/outputs").json()
                by_name = {o["name"]: o for o in listing.get("outputs", [])}
                for index, name in enumerate(saved):
                    record = by_name.get(name)
                    if record is None:
                        failures.append(f"batch: {name} is not on the shelf")
                    elif record.get("prompt") != BATCH_TEXTS[index]:
                        failures.append(f"batch: {name} holds the wrong caption")

    # The progress endpoint must answer and report idle between runs.
    progress = client.get(f"{base}/api/generate/progress").json()
    if progress.get("active"):
        failures.append("progress still active after generation finished")

    # A traversal attempt must not resolve to a file outside outputs/.
    guard = client.get(f"{base}/api/outputs/..%2Fserver.py")
    if guard.status_code not in (400, 404):
        failures.append(f"traversal guard returned {guard.status_code}")

    client.close()
    print()
    if failures:
        print(f"FAILED: {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

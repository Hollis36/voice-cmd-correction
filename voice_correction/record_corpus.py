"""Interactive recorder for the real-audio voice corpus.

Usage:
    uv run python -m research.voice_correction.record_corpus \\
        --n 50 --speaker you --out-dir research/voice_correction/audio

For each generated prompt it:
  1. Shows the prompt in big bold terminal text.
  2. Waits for ENTER (you ready).
  3. Records 4 s of mic audio.
  4. Plays it back for a 1 s sanity check (optional).
  5. Asks: [Enter] next / [r] redo / [s] skip / [q] quit.
  6. Saves WAV + appends a manifest row.

Output layout:
    <out-dir>/<speaker>/<nnn>_<timestamp>.wav
    <out-dir>/manifest.csv       (clean_prompt, wav_path, speaker, ts)

Run again with the same speaker id to resume where you left off (it
skips prompts already in the manifest).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import random
import sys
import time
from pathlib import Path

import numpy as np

from voice_correction.synth_dataset import (
    CLEAN_TEMPLATES,
    instantiate_template,
)

# ---- mic / audio I/O (lifted from demo_live.voice to avoid coupling) ----
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_SEC = 4.0


def _import_audio():
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        print("Missing dep:", exc)
        print("Install with:  uv pip install sounddevice soundfile")
        sys.exit(1)
    return sd, sf


def generate_prompts(n: int, seed: int) -> list[str]:
    """Sample N unique-ish clean commands from the grammar templates."""
    rng = random.Random(seed)
    prompts: list[str] = []
    seen: set[str] = set()
    # If we can't hit exactly n unique, allow a few duplicates at the end.
    while len(prompts) < n and len(seen) < n * 3:
        p = instantiate_template(rng.choice(CLEAN_TEMPLATES), rng)
        if p in seen and len(seen) < len(CLEAN_TEMPLATES) * 4:
            continue
        prompts.append(p)
        seen.add(p)
    return prompts


def load_done_prompts(manifest: Path, speaker: str) -> set[str]:
    """Skip any prompt we already captured for this speaker in an earlier session."""
    if not manifest.exists():
        return set()
    done: set[str] = set()
    with manifest.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("speaker") == speaker:
                done.add(row["clean_prompt"])
    return done


def append_manifest(manifest: Path, row: dict) -> None:
    exists = manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["clean_prompt", "wav_path", "speaker", "timestamp"]
        )
        if not exists:
            w.writeheader()
        w.writerow(row)


def _banner(text: str) -> None:
    line = "=" * max(60, len(text) + 8)
    print(f"\n{line}\n  {text}\n{line}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--n", type=int, default=50, help="How many prompts to capture.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--speaker", required=True,
                    help="Short id (e.g. `you`, `alice`). Used for folder + manifest.")
    ap.add_argument("--out-dir", type=Path, default=here / "audio")
    ap.add_argument("--duration", type=float, default=DURATION_SEC)
    ap.add_argument("--no-playback", action="store_true",
                    help="Skip the 1 s playback sanity check.")
    args = ap.parse_args(argv)

    sd, sf = _import_audio()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spk_dir = args.out_dir / args.speaker
    spk_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.csv"

    prompts = generate_prompts(args.n, args.seed)
    done = load_done_prompts(manifest, args.speaker)
    todo = [p for p in prompts if p not in done]
    if not todo:
        print("Nothing to do — all prompts already recorded for this speaker.")
        return 0

    print(f"Speaker: {args.speaker}")
    print(f"Prompts to record: {len(todo)} (of {len(prompts)} total; "
          f"{len(done)} already done)")
    print(f"Each recording is {args.duration} s.")
    print("Press ENTER after each prompt to START recording.")
    print("After recording: ENTER=next, r=redo, s=skip, q=quit.\n")

    idx = 0
    while idx < len(todo):
        prompt = todo[idx]
        _banner(f"[{idx + 1}/{len(todo)}]  {prompt.upper()}")
        print("  Press ENTER when ready, or `q` to quit.")
        ans = input("> ").strip().lower()
        if ans == "q":
            print("Quitting; progress saved.")
            return 0

        # Countdown to cue the speaker.
        for k in (3, 2, 1):
            print(f"  recording in {k}...", end="\r", flush=True)
            time.sleep(0.7)
        print("  REC ● " + " " * 30)

        audio = sd.rec(
            frames=int(args.duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
        )
        sd.wait()
        peak = float(np.max(np.abs(audio))) / 32768.0
        rms = float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)))
        print(f"  done.  peak={peak:.2f}  rms={rms:.3f}")

        if not args.no_playback:
            try:
                sd.play(audio, SAMPLE_RATE)
                sd.wait()
            except Exception as exc:
                print(f"  (playback skipped: {exc})")

        action = input("  [ENTER=next, r=redo, s=skip, q=quit]: ").strip().lower()
        if action == "q":
            print("Quitting; progress saved.")
            return 0
        if action == "r":
            continue
        if action == "s":
            idx += 1
            continue

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_name = f"{idx + 1:03d}_{ts}.wav"
        wav_path = spk_dir / wav_name
        sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="PCM_16")
        append_manifest(manifest, {
            "clean_prompt": prompt,
            "wav_path": str(wav_path.relative_to(args.out_dir)),
            "speaker": args.speaker,
            "timestamp": ts,
        })
        idx += 1

    print(f"\nDone. {idx} utterances captured for speaker {args.speaker!r}.")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

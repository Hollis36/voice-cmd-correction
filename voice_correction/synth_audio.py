"""Synthesize speech audio via macOS `say` command to stand in for real speakers.

Rationale: zero-cost "multi-speaker" corpus. Ten built-in voices produce
distinct acoustic profiles that, while cleaner than human speech, still
expose ASR prosody/phonetic errors — particularly when combined with
controlled additive noise.

We consider this an *auxiliary* dataset, not a substitute for real audio.
It's honest enough to:
  - validate the pipeline end-to-end,
  - stress-test the correctors under known noise levels,
  - generate a reproducible multi-condition dataset without recording,
but not honest enough to claim it matches human-speech ASR behavior.
See PAPER_DRAFT.md §6 Threats to Validity.

Usage:
    uv run python -m research.voice_correction.synth_audio --n 50
    uv run python -m research.voice_correction.synth_audio --n 50 \\
        --voices "Samantha,Alex,Karen,Daniel,Tom" \\
        --noise-snr 10,20,inf

Output:
    <audio-dir>/<voice>_<condition>/<nnn>.wav
    <audio-dir>/manifest.csv  (compatible with transcribe_corpus.py)
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from voice_correction.synth_dataset import (
    CLEAN_TEMPLATES,
    instantiate_template,
)

# macOS voices known to exist on most recent systems. The script filters
# via `say -v "?"` to only use ones actually installed.
DEFAULT_VOICES: tuple[str, ...] = (
    "Samantha",   # US English female
    "Karen",      # Australian English female
    "Daniel",     # UK English male
    "Moira",      # Irish English female
    "Tessa",      # South African English female
    "Ralph",      # US English male
    "Rishi",      # Indian English male (accent variety)
    "Kathy",      # US English female (different timbre from Samantha)
)


def available_voices() -> set[str]:
    """Parse `say -v "?"` and return the set of English-locale voice names.

    Each line looks like:
        Samantha            en_US    # Hello! My name is Samantha.
    Name may include spaces and parens, e.g.:
        Eddy (English (UK)) en_GB    # Hello! ...
    We scan each line's tokens for the first en_* locale token and take
    everything before it as the voice name.
    """
    try:
        out = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    names: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        for i, tok in enumerate(parts):
            if tok.startswith(("en_US", "en_GB", "en_AU", "en_IE", "en_ZA", "en_IN")):
                name = " ".join(parts[:i]).strip()
                if name:
                    names.add(name)
                break
    return names


def synthesize(voice: str, text: str, out_wav: Path) -> bool:
    """Run `say` to produce a 16 kHz mono WAV. Returns False on failure."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["say", "-v", voice, "-o", str(out_wav),
             "--data-format=LEI16@16000", "--file-format=WAVE",
             text],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  synth FAILED voice={voice!r} err={e.stderr.decode()[:100]}")
        return False
    return out_wav.exists() and out_wav.stat().st_size > 44


def inject_noise(wav_path: Path, snr_db: float) -> None:
    """Add Gaussian white noise to a WAV in-place at the target SNR (dB).

    snr_db == +inf → no noise added (the caller can skip us).
    """
    if not math.isfinite(snr_db):
        return
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    signal_power = float(np.mean(pcm ** 2)) + 1e-12
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    rng = np.random.default_rng(seed=hash(str(wav_path)) & 0xFFFFFFFF)
    noise = rng.normal(0.0, math.sqrt(target_noise_power), size=pcm.shape)
    noisy = np.clip(pcm + noise, -1.0, 1.0)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(n_ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((noisy * 32767).astype(np.int16).tobytes())


def generate_prompts(n: int, seed: int) -> list[str]:
    """Reuse the Week 1 grammar sampler."""
    import random as _r
    rng = _r.Random(seed)
    out: list[str] = []
    while len(out) < n:
        out.append(instantiate_template(rng.choice(CLEAN_TEMPLATES), rng))
    return out


def snr_tag(snr_db: float) -> str:
    if not math.isfinite(snr_db):
        return "clean"
    return f"snr{int(snr_db)}"


def main(argv: list[str]) -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50,
                    help="Prompts per voice per SNR condition.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--voices", type=str,
                    default=",".join(DEFAULT_VOICES),
                    help="Comma-separated macOS voice names.")
    ap.add_argument("--noise-snr", type=str, default="inf",
                    help="Comma list of SNR dB; `inf` = no noise. "
                         "Example: '10,20,inf' → 3 conditions per voice.")
    ap.add_argument("--out-dir", type=Path,
                    default=here / "audio_tts")
    args = ap.parse_args(argv)

    if sys.platform != "darwin":
        print("This script uses macOS `say`. Non-darwin systems need an alternative "
              "(piper-tts, coqui, openai TTS).")
        return 1

    installed = available_voices()
    requested = [v.strip() for v in args.voices.split(",") if v.strip()]
    voices = [v for v in requested if v in installed]
    missing = [v for v in requested if v not in installed]
    if missing:
        print(f"WARN: voices not installed on this Mac, skipping: {missing}")
    if not voices:
        print("No requested voices are installed. Available English voices:")
        for v in sorted(installed):
            print(f"  {v}")
        return 1

    snr_tokens = [s.strip().lower() for s in args.noise_snr.split(",")]
    snr_values: list[float] = []
    for t in snr_tokens:
        snr_values.append(float("inf") if t in ("inf", "clean", "none") else float(t))

    prompts = generate_prompts(args.n, args.seed)

    print(f"Voices:     {voices}")
    print(f"Conditions: {[snr_tag(s) for s in snr_values]}")
    print(f"Prompts:    {args.n}")
    total = len(voices) * len(snr_values) * args.n
    print(f"Total utterances to synthesize: {total}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    rows_done: set[tuple[str, str, str]] = set()   # (speaker, wav_path, prompt)
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows_done.add((r["speaker"], r["wav_path"], r["clean_prompt"]))

    written = 0
    with manifest_path.open(
        "a" if manifest_path.exists() else "w",
        newline="", encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clean_prompt", "wav_path", "speaker", "timestamp"],
        )
        if not rows_done:
            writer.writeheader()

        for voice in voices:
            for snr in snr_values:
                tag = f"{voice.replace(' ', '_')}_{snr_tag(snr)}"
                for i, prompt in enumerate(prompts, start=1):
                    wav_name = f"{i:03d}.wav"
                    rel_path = f"{tag}/{wav_name}"
                    speaker_id = f"tts_{tag}"
                    if (speaker_id, rel_path, prompt) in rows_done:
                        continue

                    wav = args.out_dir / rel_path
                    ok = synthesize(voice, prompt, wav)
                    if not ok:
                        continue
                    inject_noise(wav, snr)
                    writer.writerow({
                        "clean_prompt": prompt,
                        "wav_path": rel_path,
                        "speaker": speaker_id,
                        "timestamp": "",
                    })
                    written += 1
                    if written % 20 == 0:
                        print(f"  … {written}/{total}")

    print(f"\nDone. {written} new WAVs written.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

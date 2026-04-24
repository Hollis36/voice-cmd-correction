"""Run two ASR backends over the recorded corpus and emit transcripts.csv.

Usage:
    uv run python -m research.voice_correction.transcribe_corpus \\
        --audio-dir research/voice_correction/audio

Backends:
  - google   — Google Web Speech (via SpeechRecognition). Needs network.
               Already used by demo_live/voice.py — no API key.
  - whisper  — faster-whisper tiny.en (CPU, ~39 MB download, Apple Silicon OK).
               Install with:  uv pip install faster-whisper

If faster-whisper is not installed, the script runs google-only and warns.

Output: `<audio-dir>/transcripts.csv` with columns
    wav_path, speaker, clean_prompt, backend, raw, latency_ms
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import wave
from pathlib import Path

import numpy as np


def _load_wav(path: Path) -> tuple[bytes, np.ndarray, int]:
    """Return (wav_bytes_with_header, float32_mono, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    wav_bytes = path.read_bytes()
    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    f32 = pcm.astype(np.float32) / 32768.0
    return wav_bytes, f32, sr


def run_google(wav_bytes: bytes, sr: int, preferred_lang: str = "en-US") -> tuple[str, float, str]:
    """Re-use the project's google transcription; return (text, ms, error)."""
    try:
        import speech_recognition as sr_mod
    except ImportError:
        return "", 0.0, "speech_recognition not installed"
    recognizer = sr_mod.Recognizer()
    audio_data = sr_mod.AudioData(wav_bytes[44:], sr, 2)
    t0 = time.perf_counter()
    try:
        text = recognizer.recognize_google(audio_data, language=preferred_lang).strip()
        ms = (time.perf_counter() - t0) * 1000
        return text, ms, ""
    except sr_mod.UnknownValueError:
        return "", (time.perf_counter() - t0) * 1000, "unrecognized"
    except sr_mod.RequestError as e:
        return "", (time.perf_counter() - t0) * 1000, f"network:{e}"


def _get_whisper_model():
    """Lazy load faster-whisper. Returns None if not installed."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    # tiny.en is ~39 MB, English-only, fast on CPU.
    return WhisperModel("tiny.en", device="cpu", compute_type="int8")


def run_whisper(model, f32_audio: np.ndarray, sr: int) -> tuple[str, float, str]:
    if model is None:
        return "", 0.0, "faster-whisper not installed"
    t0 = time.perf_counter()
    segments, _info = model.transcribe(f32_audio, language="en", beam_size=1)
    text = " ".join(s.text for s in segments).strip()
    ms = (time.perf_counter() - t0) * 1000
    return text, ms, ""


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--audio-dir", type=Path, default=here / "audio")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Defaults to <audio-dir>/manifest.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to <audio-dir>/transcripts.csv")
    ap.add_argument("--backends", default="google,whisper",
                    help="Comma list: google,whisper")
    args = ap.parse_args(argv)

    manifest = args.manifest or args.audio_dir / "manifest.csv"
    out_path = args.out or args.audio_dir / "transcripts.csv"
    if not manifest.exists():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 1

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    whisper_model = _get_whisper_model() if "whisper" in backends else None
    if "whisper" in backends and whisper_model is None:
        print("WARN: faster-whisper not installed — skipping whisper backend.")
        print("      install with:  uv pip install faster-whisper")
        backends = [b for b in backends if b != "whisper"]

    rows_in: list[dict] = []
    with manifest.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows_in.append(r)
    print(f"Transcribing {len(rows_in)} utterances x {len(backends)} backends "
          f"= {len(rows_in) * len(backends)} calls.")

    out_rows: list[dict] = []
    for i, r in enumerate(rows_in, 1):
        wav = args.audio_dir / r["wav_path"]
        if not wav.exists():
            print(f"  [{i}] MISSING {wav}")
            continue
        wav_bytes, f32, sr = _load_wav(wav)
        print(f"  [{i}/{len(rows_in)}] {r['speaker']:10s} {wav.name}")
        for b in backends:
            if b == "google":
                text, ms, err = run_google(wav_bytes, sr)
            elif b == "whisper":
                text, ms, err = run_whisper(whisper_model, f32, sr)
            else:
                continue
            tag = f"  -> {b:8s} [{ms:6.0f} ms]"
            print(f"{tag} {text!r}  {err}")
            out_rows.append({
                "wav_path": r["wav_path"],
                "speaker": r["speaker"],
                "clean_prompt": r["clean_prompt"],
                "backend": b,
                "raw": text,
                "latency_ms": f"{ms:.1f}",
                "error": err,
            })

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["wav_path", "speaker", "clean_prompt",
                        "backend", "raw", "latency_ms", "error"],
        )
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

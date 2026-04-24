# voice-cmd-correction

**Cheap, Robust Correction of Noisy Voice Commands for Restricted-Vocabulary Robotics**

Companion code for the workshop paper submitted to CoRL 2026.

## Reproduce in < 15 seconds

```bash
pip install "voice-cmd-correction[figures] @ git+https://github.com/Hollis36/voice-cmd-correction"

# Synthetic benchmark (Table 1 + B6 extended)
python -m voice_correction.eval_with_b6

# Held-out generalization (Table 2)
python -m voice_correction.held_out_eval

# Figures
python -m voice_correction.plot_results   # fig01, fig02
python -m voice_correction.plot_heldout   # fig03
python -m voice_correction.plot_tts       # fig04
```

No GPU. No network (except first pip install). Runs on CPU in < 15 s.

## Key results

| Method | In-dist CmdAcc | Held-out CmdAcc | Whisper TTS CmdAcc |
|--------|---------------|-----------------|-------------------|
| B2 edit-distance | 0.383 | 0.383 | 0.62 |
| B3 hand-phonetic | 0.478 | 0.475 | 0.67 |
| B5 filter+learned | 0.708 | 0.573 | 0.62 |
| **B6 bigram reranker** | **0.603** | **0.585** | **0.70** |

B6 closes the generalization gap from −13.5 pp (B5) to −1.8 pp at ~100 µs per utterance on CPU.

## Methods (B0–B6)

| ID | Method |
|----|--------|
| B0 | Identity (no correction) |
| B1 | Strict vocab filter |
| B2 | Edit-distance closest match |
| B3 | Hand-curated phonetic table + B2 |
| B4 | Learned phonetic table + B2 |
| B5 | Filler-word filter + B4 |
| B6 | B5 pipeline + bigram LM reranker |

## License

Apache 2.0

# Spec S5.1 — Post-Processing Corrections

## Overview
Add a post-processing step after Whisper's EnglishTextNormalizer to fix common ASR errors in children's speech transcription. This targets the final WER squeeze by correcting systematic mistakes the model makes.

## Depends On
- S3.3 (Ensemble inference pipeline) — done

## Location
- `src/utils.py` — post-processing functions
- `submission/main.py` — integrate post-processing into inference pipeline

## Requirements

### R1: Common ASR Error Corrections
A dictionary-based replacement system that fixes known Whisper transcription errors on children's speech:
- Repeated words from hallucination (e.g., "the the the" → "the")
- Common homophone/near-miss errors specific to children's speech
- Artifact tokens Whisper sometimes emits (e.g., "[inaudible]", "(laughing)", "♪", "...")
- Leading/trailing whitespace artifacts

### R2: Children's Vocabulary Spell-Check
A lightweight spell-correction layer using a curated word list:
- Target common child speech patterns that Whisper misspells (phonetic approximations)
- Do NOT "correct" valid child speech forms like "goed", "tooths", "bestest" — these are real utterances
- Only fix clear ASR artifacts, not child language patterns
- Must be fast (< 1ms per utterance) — no external API calls

### R3: Hallucination Detection & Cleanup
- Detect repeated n-grams (same word repeated 3+ times consecutively) and collapse to single occurrence
- Detect predictions that are suspiciously long relative to audio duration (if available)
- Strip common Whisper hallucination artifacts: music symbols, bracketed annotations, etc.

### R4: Integration with Inference Pipeline
- Post-processing runs AFTER EnglishTextNormalizer
- Applied in `submission/main.py` via a single function call
- Must not break existing normalize_text behavior
- New function: `postprocess_text(text: str) -> str`
- New function: `normalize_and_postprocess(text: str) -> str` combining both steps

## Outcomes
- All predictions pass through post-processing before output
- Measurable WER improvement on validation set (target: 0.5-2% absolute reduction)
- No regression on clean predictions
- < 1ms latency per utterance

## TDD Notes
- Test each correction category independently
- Test that valid child speech forms are preserved
- Test idempotency (applying twice gives same result)
- Test empty/None/whitespace inputs
- Test integration with normalize_text
- Mock no external services (pure string processing)

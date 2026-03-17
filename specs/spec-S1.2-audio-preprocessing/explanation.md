# Explanation S1.2 — Audio Preprocessing Pipeline

## Why
The audio preprocessing pipeline is the foundation for all downstream processing in ChildWhisper. Raw competition audio comes in varying sample rates and formats. Before any model can process it, audio must be standardized to 16 kHz mono (Whisper's expected input format). Additionally, silent clips, too-short clips, and too-long clips need to be filtered to avoid wasting compute and introducing noise into training.

## What
`src/preprocess.py` provides six public functions:

1. **`load_audio`** — Loads any audio file (FLAC/WAV) via librosa, resamples to 16 kHz mono
2. **`trim_silence`** — Removes leading/trailing silence using librosa's trim with top_db=30
3. **`is_silence`** — Detects if an entire clip is silence by computing RMS energy in dB against a -40 dB threshold
4. **`get_duration`** / **`is_valid_duration`** — Duration computation and 0.3–30s range validation
5. **`preprocess_utterance`** — Full pipeline: load → trim → silence check → duration filter → return structured dict or None
6. **`load_metadata`** — Parses JSONL metadata files into list of dicts

## How
- **librosa** handles all audio I/O and resampling — chosen for reliability and consistency with the competition ecosystem
- **RMS-based silence detection** uses `20 * log10(rms)` with a 1e-10 epsilon to avoid log(0). The -40 dB threshold matches `configs/training_config.yaml`
- **`preprocess_utterance`** short-circuits early: empty transcripts are rejected before any audio I/O. Silent clips return a dict with empty transcript (not None) so downstream can distinguish "silence detected" from "invalid sample"
- All default parameters match values in `training_config.yaml`

## Connections
- **S1.1** (Project structure) — provides configs/training_config.yaml with preprocessing constants
- **S1.3** (Text normalization) — will normalize transcripts; preprocess.py handles raw transcripts
- **S1.4** (Inference pipeline) — will use `load_audio` and `is_silence` for test-time preprocessing
- **S1.5** (Local validation) — will use `load_metadata` to load train/val splits
- **S2.1** (Dataset class) — will call `preprocess_utterance` to build the training dataset

## Test Coverage
23 tests covering all public functions, edge cases (empty audio, boundary durations, near-threshold silence), and both str/Path input types. All external I/O (librosa) is mocked.

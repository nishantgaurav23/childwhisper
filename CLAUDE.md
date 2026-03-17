# ChildWhisper — Pasketti Children's ASR Challenge (Word Track)

Two-model ensemble (Whisper-large-v3 LoRA + Whisper-small full fine-tune) for children's speech transcription. Trained on competition data + classroom noise augmentation, deployed on A100 inference runtime. Target WER: <0.20. Budget: $0-10/month.

## Key Rules

- **NEVER** write code without a spec — always run `/create-spec` first
- **NEVER** skip TDD — write tests FIRST, then implement (Red -> Green -> Refactor)
- **NEVER** hardcode API keys or secrets — all config via `.env` files
- **NEVER** hit real external APIs in tests — mock all external services
- **NEVER** depend on paid services for core functionality — Kaggle free tier + Colab free tier + HuggingFace Hub free
- **ALWAYS** update `roadmap.md` status after completing a spec
- **ALWAYS** run `/explain-spec` after completing a spec
- **ALWAYS** run `ruff` for linting (line-length: 100)
- **ALWAYS** apply Whisper `EnglishTextNormalizer` to all predictions before output
- **ALWAYS** resample audio to 16 kHz mono before processing
- **ALWAYS** validate by `child_id` split (no speaker leakage between train/val)
- **ALWAYS** test inference code on MacBook (MPS/CPU) before Kaggle submission
- **ALWAYS** checkpoint to HuggingFace Hub every 500 steps (Kaggle sessions are ephemeral)

## Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| Language | Python 3.11 | Free |
| ASR Framework | HuggingFace Transformers + PEFT | Free |
| Primary Model | Whisper-large-v3 + LoRA (r=32) | Free (open weights) |
| Secondary Model | Whisper-small (full fine-tune) | Free (open weights) |
| Quantization | bitsandbytes (INT8 for training) | Free |
| Audio Processing | librosa + soundfile + torchaudio | Free |
| Augmentation | audiomentations | Free |
| Evaluation | jiwer (WER computation) | Free |
| Dev Machine | MacBook (Apple Silicon, MPS backend) | $0 |
| GPU Training | Kaggle T4 (30 hrs/wk) + Colab T4 | $0 |
| Model Storage | HuggingFace Hub (private repos) | $0 |
| Inference Runtime | A100 80GB, 2hr limit, offline | Competition |
| Testing | pytest | Free |
| Linting | ruff (line-length: 100) | Free |

## Project Structure

```
childwhisper/
├── CLAUDE.md                     # This file
├── roadmap.md                    # Master development plan (source of truth)
├── design.md                     # Architecture document
├── requirements.md               # Requirements specification
├── requirements.txt              # Python dependencies
├── .claude/
│   ├── CLAUDE.md                 # Claude Code context
│   └── commands/                 # Spec-driven dev commands
├── specs/                        # Spec folders
│   └── spec-S{x}.{y}-{slug}/
│       ├── spec.md
│       ├── checklist.md
│       └── explanation.md
├── src/
│   ├── preprocess.py             # Data loading & preprocessing
│   ├── augment.py                # Noise augmentation pipeline
│   ├── dataset.py                # PyTorch Dataset class
│   ├── train_whisper_small.py    # Full fine-tune script
│   ├── train_whisper_lora.py     # LoRA fine-tune script
│   ├── evaluate.py               # WER evaluation with normalizer
│   └── utils.py                  # Shared utilities
├── notebooks/
│   ├── 01_eda.ipynb              # Data exploration (MacBook)
│   ├── 02_train_small.ipynb      # Kaggle training notebook
│   ├── 03_train_lora.ipynb       # Kaggle training notebook
│   └── 04_augmented.ipynb        # Kaggle augmented training
├── submission/
│   ├── main.py                   # Inference entrypoint (A100)
│   ├── model_weights/            # Downloaded from HF Hub
│   └── utils/
├── tests/
│   ├── test_preprocess.py
│   ├── test_inference.py
│   ├── test_dataset.py
│   └── test_submission.py
├── scripts/
│   ├── download_data.sh          # Download competition data
│   ├── download_weights.sh       # Pull weights from HF Hub
│   └── build_submission.sh       # Package submission.zip
├── configs/
│   └── training_config.yaml
└── data/                         # .gitignored
    ├── audio_sample/             # ~100 FLAC files for local testing
    └── train_word_transcripts.jsonl
```

## Spec Folder Convention

```
specs/spec-S{phase}.{number}-{slug}/
  spec.md        <- Requirements, outcomes, TDD notes
  checklist.md   <- Phase-by-phase implementation tracker
  explanation.md <- Post-completion: why, what, how, connections
```

## Spec-Driven Development Commands

| Command | Input | Action |
|---------|-------|--------|
| `/create-spec` | spec ID + slug | Create spec.md + checklist.md from roadmap |
| `/implement-spec` | spec ID | TDD implementation (Red -> Green -> Refactor) |
| `/verify-spec` | spec ID | Post-implementation audit (tests, lint, outcomes) |
| `/check-spec-deps` | spec ID | Verify all prerequisite specs are done |
| `/start-spec-dev` | spec ID | Full workflow: check deps -> create spec -> implement -> verify -> explain |
| `/explain-spec` | spec ID | Generate explanation.md: why, what, how, connections |

## Status Flow

```
pending -> spec-written -> done
```

## Code Standards

### Audio Processing
- All audio resampled to 16 kHz mono via librosa
- Trim silence with `librosa.effects.trim(top_db=30)`
- Duration filter: skip < 0.3s or > 30s
- Silence detection: RMS energy < -40 dB -> return empty string

### Model Training
- Whisper-large-v3: LoRA r=32, alpha=64, target q_proj + v_proj, INT8 quantization
- Whisper-small: full fine-tune with gradient checkpointing
- SpecAugment ON for both models (mask_time_prob=0.05, mask_feature_prob=0.04)
- Early stopping on validation WER
- Save checkpoints to HuggingFace Hub (private repos)

### Inference
- Sort utterances by duration (longest first) for efficient batching
- Beam search: num_beams=5
- Run both models sequentially on A100 (large first, small if time permits)
- Confidence-based merging: prefer large model, fallback to small on empty predictions
- Apply EnglishTextNormalizer to all output

### Validation
- 90/10 split by child_id (no speaker leakage)
- Stratify by age_bucket
- Track per-age-bucket WER
- Synthetic noisy validation: val audio + RealClass noise at SNR 10 dB

### Testing
- pytest for all tests
- Mock audio I/O and model inference in unit tests
- Test with small audio samples locally
- Test submission format compliance
- Target >80% coverage per spec

### Dependencies
- Pin all dependencies in `requirements.txt`
- All deps must be free / open-source
- Submission must use only pre-installed runtime packages

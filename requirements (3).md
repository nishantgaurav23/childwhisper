# Requirements — Pasketti Children's ASR Challenge (Word Track)

## 1. Problem Statement

Build an ASR model that transcribes children's speech from short FLAC audio clips. Submit as a code-execution package (no API calls allowed). Scored by Word Error Rate (WER) with a secondary Noisy WER for classroom environments.

Competition: [DrivenData — On Top of Pasketti Word Track](https://www.drivendata.org/competitions/308/childrens-word-asr/)
Deadline: ~Early April 2026
Prize Pool: $70,000 (Word Track) + $20,000 (Noisy WER bonuses)
Submissions: 3 per rolling 7-day window


## 2. Functional Requirements

### FR-1: Audio Ingestion
- Read FLAC audio files referenced by `utterance_metadata.jsonl`
- Handle variable durations (sub-second to 30+ seconds)
- Resample all audio to 16 kHz mono (test data is already 16 kHz mono; training data varies)
- Gracefully handle edge cases: silence-only clips, very short clips (<0.5s), corrupted files

### FR-2: Transcription
- Produce normalized English orthographic text for each utterance
- Handle children ages 3–12+ with speech sound disorders, non-standard grammar ("goed", "tooths"), and occasional single non-English words embedded in English
- Do NOT transcribe environmental noise, non-lexical sounds, or disfluencies (per competition rules)

### FR-3: Text Normalization
- Apply Whisper's `EnglishTextNormalizer` to all predictions before output
- This normalizer handles: lowercase, contraction expansion, number standardization, punctuation removal, whitespace normalization, diacritics removal

### FR-4: Output Format
- Write JSONL to `/code_execution/submission/submission.jsonl`
- Each line: `{"utterance_id": "<id>", "orthographic_text": "<prediction>"}`
- One line per utterance, no missing IDs

### FR-5: Noise Robustness
- Must generalize to classroom environments with background noise (children talking, HVAC, furniture), crosstalk, and reverberation
- Target SNR range: 0–20 dB
- This subset is evaluated separately for the Noisy WER bonus


## 3. Non-Functional Requirements

### NFR-1: Inference Runtime Constraints
| Resource         | Limit              |
|------------------|--------------------|
| GPU              | NVIDIA A100, 80 GB |
| CPU              | 24 vCPUs           |
| RAM              | 220 GB             |
| Wall time        | 2 hours            |
| Network          | None (offline)     |
| Log output       | 500 lines max      |

### NFR-2: Submission Package
- Single `submission.zip` with `main.py` at root
- All model weights bundled inside (no downloads at runtime)
- Must run without errors in the competition Docker container
- Pre-installed: Python 3.11, CUDA 12.6, PyTorch ≥2.9, transformers ≥4.52.4, peft, nemo_toolkit[asr] ≥2.5, librosa, soundfile, torchaudio

### NFR-3: Budget Constraint
- Monthly spend: **$5–10 maximum**
- Primary dev machine: MacBook (Apple Silicon M-series)
- GPU training: Kaggle free tier (30 hrs/wk, 2× T4 16 GB) + Colab free tier (T4)
- One-time boost: GCP $300 new-user credits (optional)
- Storage: HuggingFace Hub (free) for model checkpoints; Kaggle datasets for audio


## 4. Data Requirements

### 4.1 Competition Data (Mandatory)
| Corpus                | Source       | Contents                          |
|-----------------------|--------------|-----------------------------------|
| DrivenData corpus     | DrivenData   | 3 audio zips + `train_word_transcripts.jsonl` |
| TalkBank corpus       | TalkBank     | 1 audio zip + `train_word_transcripts.jsonl`  |
| RealClass noise       | DrivenData   | Synthetic classroom background noise          |

TalkBank requires a free account + data access request form.

### 4.2 JSONL Schema (Training)
```json
{
  "utterance_id": "string",
  "child_id": "string",
  "session_id": "string",
  "audio_path": "audio/{utterance_id}.flac",
  "audio_duration_sec": 3.42,
  "age_bucket": "5-7",
  "md5_hash": "string",
  "filesize_bytes": 54321,
  "orthographic_text": "i goed to the store"
}
```

### 4.3 External Data (Optional, License-Safe)
| Dataset     | Size     | Use Case                     | License    |
|-------------|----------|------------------------------|------------|
| MUSAN       | ~42 hrs  | Noise augmentation (babble, music, ambient) | CC-BY 4.0 |
| CHILDES/OCSC| ~156 hrs | Additional child speech       | CC-BY      |
| LibriSpeech | 960 hrs  | General English regularization| CC-BY 4.0  |
| OpenSLR RIRs| —        | Room impulse responses        | Apache 2.0 |

**Avoid**: MyST corpus (CC BY-NC-SA — not prize-eligible).


## 5. Development Environment

### 5.1 MacBook (Local — CPU Only)
Used for: data exploration, preprocessing pipeline, inference code, debugging, unit tests.

```
Python:      3.11
OS:          macOS (Apple Silicon M1/M2/M3)
Key libs:    torch (MPS backend for light testing), transformers, peft,
             librosa, soundfile, audiomentations, jiwer, pandas
Storage:     ~50 GB for a subset of training audio + model weights
```

**Critical**: All training code must be written to be hardware-agnostic — develop on Mac, run training on Kaggle/Colab by switching device and paths only.

### 5.2 Kaggle Notebooks (GPU Training — Primary)
```
GPU:         2× NVIDIA T4 (16 GB each)
Quota:       30 GPU hours/week
Disk:        ~73 GB ephemeral
Persistence: Save checkpoints to HuggingFace Hub
```

### 5.3 Google Colab (GPU Training — Secondary)
```
GPU:         1× T4 (16 GB), occasionally V100
Quota:       ~15–30 hrs/week (variable, free tier)
Disk:        ~100 GB ephemeral
```

### 5.4 Submission Testing
- Clone `github.com/drivendataorg/childrens-speech-recognition-runtime`
- Build Docker image locally (requires Docker Desktop)
- Run `make test-submission` with a small audio sample


## 6. Python Dependencies

### 6.1 Core (Training & Inference)
```
torch>=2.1
torchaudio>=2.1
transformers>=4.44
peft>=0.12
accelerate>=0.33
bitsandbytes>=0.43       # 8-bit quantization + optimizer
datasets>=2.20
librosa>=0.10
soundfile>=0.12
jiwer>=3.0               # WER computation
```

### 6.2 Augmentation
```
audiomentations>=0.36
```

### 6.3 Inference Optimization (Optional)
```
faster-whisper>=1.0       # CTranslate2 backend, 4-10× speedup
```

### 6.4 Dev & Testing
```
jupyter
pandas
matplotlib
tqdm
pytest
```


## 7. Model Requirements

### 7.1 Primary Model: Whisper-large-v3 + LoRA
- Base: `openai/whisper-large-v3` (1.55B params, ~3 GB weights)
- Adapter: LoRA r=32, alpha=64, targeting q_proj + v_proj
- Trainable params: ~15M (~1% of base)
- Adapter checkpoint: ~63 MB
- Training VRAM: ~8 GB (LoRA + INT8 quantization on T4)
- Inference VRAM: ~6 GB (fp16 on A100, no quantization needed)

### 7.2 Secondary Model: Whisper-small (Full Fine-Tune)
- Base: `openai/whisper-small` (242M params, ~0.5 GB weights)
- Full fine-tuning with gradient checkpointing
- Training VRAM: ~6 GB on T4
- Serves as fast iteration model + ensemble candidate

### 7.3 Stretch Goal: Parakeet TDT 0.6B + NeMo Adapters
- Pre-installed in runtime via nemo_toolkit
- Best published fine-tuned WER on child speech (8.5% on MyST)
- Adapter training via NeMo's native adapter framework
- Consider if Kaggle GPU hours allow

### 7.4 Submission Size Budget
| Component                      | Size (approx) |
|--------------------------------|---------------|
| Whisper-large-v3 base weights  | ~3.1 GB       |
| LoRA adapter                   | ~63 MB        |
| Whisper-small full weights     | ~0.5 GB       |
| Code + utils                   | <1 MB         |
| **Total**                      | **~3.7 GB**   |

DrivenData zip upload limit should accommodate this; verify with runtime docs.


## 8. Evaluation & Validation

### 8.1 Primary Metric
**WER (Word Error Rate)** = (Substitutions + Deletions + Insertions) / Reference Word Count
Lower is better. Computed after Whisper English text normalization on both sides.

### 8.2 Secondary Metric
**Noisy WER** — WER on classroom-environment subset only.

### 8.3 Local Validation Strategy
- 90/10 stratified split by `child_id` (no child appears in both train and val)
- Stratify by `age_bucket` to ensure representation across age groups
- Compute WER using `jiwer` library with Whisper normalizer applied
- Track per-age-bucket WER to identify weak spots
- Create a synthetic "noisy validation" set by mixing val audio with RealClass noise at SNR 5–15 dB


## 9. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Kaggle GPU quota exhausted mid-week | Training stalls | Checkpoint every 500 steps to HF Hub; use Colab as backup |
| Submission zip too large | Cannot submit | Use LoRA-only adapters; share base weights via runtime pre-installed packages |
| Inference exceeds 2-hour limit | Failed submission | Profile locally with full test set size estimate; use faster-whisper or reduce beam size |
| Overfitting to competition data | High WER on unseen test | Hold out 10% for validation; use augmentation; mix in external adult speech |
| Model hallucinates on silence | WER spike | Detect silence via energy threshold; return empty string for silent clips |
| T4 OOM during training | Training crash | Reduce batch size to 1; increase gradient accumulation; verify INT8 loading |

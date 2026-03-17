# Design — Pasketti Children's ASR Challenge (Word Track)

## 1. Solution Overview

A two-model ensemble for children's speech transcription: **Whisper-large-v3 fine-tuned with LoRA** as the primary model and **Whisper-small fully fine-tuned** as a fast secondary model. Both are trained on competition data augmented with classroom noise, deployed sequentially on the A100 inference runtime, with predictions merged via confidence-based selection.

**Target WER**: <0.20 (beating the 0.2370 benchmark)
**Budget**: $5–10/month (Kaggle free tier + Colab + optional minimal cloud spend)
**Dev Machine**: MacBook (Apple Silicon) for code development; Kaggle/Colab T4 for training


## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE (Kaggle/Colab T4)          │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────────────────┐ │
│  │ Raw FLAC │──▶│ Preprocessor │──▶│ Augmented Dataset        │ │
│  │ + JSONL  │   │ (16kHz mono) │   │ (clean + RealClass mix)  │ │
│  └──────────┘   └──────────────┘   └────────┬────────────────┘ │
│                                              │                  │
│                        ┌─────────────────────┼────────────┐     │
│                        ▼                     ▼            │     │
│              ┌──────────────────┐  ┌──────────────────┐   │     │
│              │ Whisper-large-v3 │  │ Whisper-small    │   │     │
│              │ + LoRA (r=32)    │  │ Full Fine-Tune   │   │     │
│              │ + INT8 quant     │  │ + Grad Ckpt      │   │     │
│              └────────┬─────────┘  └────────┬─────────┘   │     │
│                       │                     │             │     │
│                       ▼                     ▼             │     │
│              ┌─────────────┐      ┌─────────────┐        │     │
│              │ LoRA Adapter│      │ Full Weights │        │     │
│              │ (~63 MB)    │      │ (~500 MB)    │        │     │
│              └─────────────┘      └─────────────┘        │     │
│                                                           │     │
│  Upload checkpoints to HuggingFace Hub ◀──────────────────┘     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 INFERENCE PIPELINE (A100, 80 GB VRAM)           │
│                                                                 │
│  ┌──────────────────┐                                           │
│  │ utterance_meta-  │                                           │
│  │ data.jsonl       │──▶ Sort by duration (longest first)       │
│  └──────────────────┘              │                            │
│                                    ▼                            │
│                    ┌───────────────────────────┐                │
│                    │ Batch Audio Loader         │                │
│                    │ (16kHz, padded batches)    │                │
│                    └──────────┬────────────────┘                │
│                               │                                 │
│              ┌────────────────┼────────────────┐                │
│              ▼                                 ▼                │
│   ┌─────────────────────┐          ┌────────────────────┐       │
│   │ Model A: Whisper-   │          │ Model B: Whisper-  │       │
│   │ large-v3 + LoRA     │          │ small (full)       │       │
│   │ fp16, beam=5        │          │ fp16, beam=5       │       │
│   └──────────┬──────────┘          └─────────┬──────────┘       │
│              │                                │                 │
│              ▼                                ▼                 │
│   ┌─────────────────────────────────────────────────┐           │
│   │ Prediction Merger                                │           │
│   │ - If both agree → use shared prediction          │           │
│   │ - If disagree → pick lower avg log-probability   │           │
│   │   (higher confidence)                            │           │
│   └──────────────────────┬──────────────────────────┘           │
│                          ▼                                      │
│               ┌─────────────────────┐                           │
│               │ Whisper EnglishText │                           │
│               │ Normalizer          │                           │
│               └──────────┬──────────┘                           │
│                          ▼                                      │
│               ┌─────────────────────┐                           │
│               │ submission.jsonl    │                           │
│               └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```


## 3. Development Phases

Development is split into 5 phases. Each phase produces a submittable solution — every phase improves on the last, so you always have something to submit even if time runs out.

### Phase 1: Zero-Shot Baseline (MacBook, 1–2 days)
**Goal**: Get a working submission with zero training. Establish baseline WER.

- Write the full inference pipeline (`main.py`) using `openai/whisper-small` zero-shot
- Build local validation: 90/10 split by `child_id`, stratified by `age_bucket`
- Implement Whisper `EnglishTextNormalizer` in the output path
- Implement silence detection (return empty string for energy < threshold)
- Test with competition runtime Docker locally
- **Expected WER**: ~0.25–0.30 (zero-shot whisper-small on children's speech)
- **Cost**: $0

### Phase 2: Whisper-small Full Fine-Tune (Kaggle, 3–4 days)
**Goal**: Fine-tune the smallest viable model end-to-end. First real improvement.

- Upload competition audio to Kaggle dataset (private)
- Preprocess: resample to 16kHz, normalize transcripts, filter 1–30s duration
- Full fine-tune whisper-small on Kaggle T4:
  - Learning rate: 1e-5, warmup 500 steps
  - Batch: per_device=2, gradient_accumulation=8 (effective=16)
  - fp16, gradient checkpointing enabled
  - SpecAugment ON (mask_time_prob=0.05, mask_feature_prob=0.04)
  - 3 epochs with early stopping on val WER
  - ~6 hours total training time
- Save checkpoint to HuggingFace Hub (private repo)
- Submit with fine-tuned whisper-small
- **Expected WER**: ~0.15–0.20
- **Cost**: $0 (Kaggle free tier)

### Phase 3: Whisper-large-v3 LoRA (Kaggle, 4–5 days)
**Goal**: Add the larger model with parameter-efficient fine-tuning.

- Load `openai/whisper-large-v3` in INT8 via `bitsandbytes`
- Apply LoRA: r=32, alpha=64, target_modules=["q_proj", "v_proj"]
- Training on Kaggle T4 (~8 GB VRAM with INT8+LoRA):
  - Learning rate: 1e-3 (higher for LoRA, frozen base)
  - Batch: per_device=1, gradient_accumulation=16
  - fp16, gradient checkpointing
  - SpecAugment ON
  - 1,500–2,000 steps with early stopping
  - ~8 hours total
- Save LoRA adapter (~63 MB) to HuggingFace Hub
- Update `main.py` to run both models as ensemble
- **Expected WER**: ~0.12–0.17
- **Cost**: $0

### Phase 4: Noise Augmentation (Kaggle/Colab, 3–4 days)
**Goal**: Improve Noisy WER for the classroom bonus prize.

- Download RealClass noise dataset + MUSAN babble/noise subsets
- Build augmentation pipeline with `audiomentations`:
  - 50% of training samples: add RealClass background noise at SNR 5–20 dB
  - 20% of training samples: add MUSAN babble noise at SNR 0–15 dB
  - 30% of training samples: clean (no augmentation)
  - All samples: SpecAugment
- Re-train whisper-large-v3 LoRA on augmented data
- Optionally re-train whisper-small on augmented data
- Validate on synthetic noisy validation set (val audio + RealClass at SNR 10 dB)
- **Expected Noisy WER improvement**: 15–30% relative reduction
- **Cost**: $0

### Phase 5: Optimizations & Polish (MacBook + Kaggle, 2–3 days)
**Goal**: Squeeze out final WER improvements before deadline.

Pick from these based on remaining time and observed error patterns:

- **Post-processing**: Simple spell-check against a children's vocabulary list; common ASR error corrections (e.g., "gonna"→"going to" if normalizer doesn't catch it)
- **Inference speed**: Convert whisper-large-v3 to CTranslate2/faster-whisper format for 4× speedup, freeing time for larger beam search or more ensemble passes
- **Pseudo-labeling**: Run best model on unlabeled or low-confidence training samples, add high-confidence predictions as extra training data, retrain
- **Test-Time Adaptation (TTA)**: Run 1-step gradient update on encoder for each test utterance using SUTA (entropy minimization) — feasible on A100 within time budget
- **Per-age-bucket analysis**: If WER is significantly worse for age 3–4, consider training a specialized adapter
- **Cost**: $0–5 (potential small Colab Pro spend if needed for extended sessions)


## 4. Data Pipeline Design

### 4.1 Preprocessing (runs on MacBook)

```python
# preprocessing.py — runs locally, outputs processed dataset

def preprocess_utterance(audio_path, transcript, target_sr=16000):
    """Standard preprocessing for one utterance."""
    # 1. Load and resample
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # 2. Trim silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)

    # 3. Duration filter
    duration = len(audio_trimmed) / target_sr
    if duration < 0.3 or duration > 30.0:
        return None  # skip

    # 4. Normalize transcript
    normalizer = EnglishTextNormalizer()
    clean_text = normalizer(transcript)
    if len(clean_text.strip()) == 0:
        return None  # skip empty transcripts

    return audio_trimmed, clean_text, duration
```

### 4.2 Augmentation Pipeline

```python
from audiomentations import Compose, OneOf, AddBackgroundNoise, Gain

def get_augmentation(noise_dir, realclass_dir):
    return Compose([
        # Classroom noise (50% probability)
        OneOf([
            AddBackgroundNoise(
                sounds_path=str(realclass_dir),
                min_snr_in_db=5, max_snr_in_db=20, p=1.0
            ),
            AddBackgroundNoise(
                sounds_path=str(noise_dir),  # MUSAN babble
                min_snr_in_db=0, max_snr_in_db=15, p=1.0
            ),
        ], p=0.5),
        # Volume variation
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.3),
    ])
```

### 4.3 Validation Split Strategy

```
Split by child_id (no speaker leakage):
├── Train: 90% of unique child_ids
└── Val:   10% of unique child_ids
    ├── Val-clean:  original audio
    └── Val-noisy:  audio + RealClass noise at SNR 10 dB
```

Stratify by `age_bucket` so each split has proportional representation of 3-4, 5-7, 8-11, 12+, and unknown.


## 5. Model Configuration Details

### 5.1 Whisper-large-v3 + LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# Load in INT8 for training on T4
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    load_in_8bit=True,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora_config)
# Trainable: ~15M params (1% of 1.55B)

# Enable SpecAugment (OFF by default!)
model.config.apply_spec_augment = True
model.config.mask_time_prob = 0.05
model.config.mask_time_length = 10
model.config.mask_feature_prob = 0.04
model.config.mask_feature_length = 10
```

### 5.2 Whisper-small Full Fine-Tune

```python
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    torch_dtype=torch.float16,
)
model.config.apply_spec_augment = True
# Full fine-tune — all 242M params trainable
# Gradient checkpointing to fit on T4
model.gradient_checkpointing_enable()
```

### 5.3 Training Arguments (Shared Template)

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=2,     # 1 for large-v3
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,      # 16 for large-v3
    learning_rate=1e-5,                 # 1e-3 for LoRA
    warmup_steps=500,
    num_train_epochs=3,
    fp16=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=50,
    push_to_hub=True,                   # persist across sessions
    hub_model_id="nishantgaurav23/pasketti-whisper-lora",
    hub_private_repo=True,
    dataloader_num_workers=2,
    optim="adamw_8bit",                 # halves optimizer memory
    gradient_checkpointing=True,
)
```


## 6. Inference Design

### 6.1 main.py Structure

```python
"""
main.py — Competition submission entrypoint.
Runs on A100 (80 GB VRAM), no network, 2-hour limit.
"""
import json, torch, time
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel
import librosa

# --- Constants ---
DATA_DIR = Path("/code_execution/data")
OUTPUT_DIR = Path("/code_execution/submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000

def load_models():
    """Load both models. ~30 seconds on A100."""
    # Model A: Whisper-large-v3 + LoRA
    processor_lg = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model_lg = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16
    ).to(DEVICE)
    model_lg = PeftModel.from_pretrained(model_lg, "./model_weights/lora_large_v3")

    # Model B: Whisper-small (full fine-tuned)
    processor_sm = WhisperProcessor.from_pretrained("openai/whisper-small")
    model_sm = WhisperForConditionalGeneration.from_pretrained(
        "./model_weights/whisper_small_ft", torch_dtype=torch.float16
    ).to(DEVICE)

    return (model_lg, processor_lg), (model_sm, processor_sm)

def transcribe_batch(model, processor, audio_batch):
    """Transcribe a batch of audio arrays."""
    inputs = processor(
        audio_batch, sampling_rate=SAMPLE_RATE,
        return_tensors="pt", padding=True
    ).input_features.to(DEVICE, dtype=torch.float16)

    with torch.no_grad():
        generated = model.generate(
            inputs, language="en", task="transcribe",
            num_beams=5, max_new_tokens=225,
            return_dict_in_generate=True, output_scores=True,
        )
    texts = processor.batch_decode(generated.sequences, skip_special_tokens=True)
    return texts

def main():
    t0 = time.time()
    normalizer = EnglishTextNormalizer()

    # Load metadata
    meta_path = DATA_DIR / "utterance_metadata.jsonl"
    utterances = [json.loads(l) for l in meta_path.read_text().strip().split("\n")]

    # Sort by duration (longest first) for efficient batching
    utterances.sort(key=lambda u: u.get("audio_duration_sec", 0), reverse=True)

    # Load models
    (model_lg, proc_lg), (model_sm, proc_sm) = load_models()

    # Inference — Model A (large-v3 + LoRA)
    predictions = {}
    BATCH_SIZE = 16
    for i in range(0, len(utterances), BATCH_SIZE):
        batch = utterances[i:i+BATCH_SIZE]
        audios = []
        for u in batch:
            audio, _ = librosa.load(DATA_DIR / u["audio_path"], sr=SAMPLE_RATE, mono=True)
            audios.append(audio)
        texts = transcribe_batch(model_lg, proc_lg, audios)
        for u, t in zip(batch, texts):
            predictions[u["utterance_id"]] = normalizer(t)

    elapsed = time.time() - t0
    print(f"Model A done in {elapsed:.0f}s")

    # If time permits (< 90 min elapsed), run Model B for ensemble
    if elapsed < 5400:
        del model_lg  # free VRAM
        torch.cuda.empty_cache()
        predictions_b = {}
        for i in range(0, len(utterances), BATCH_SIZE):
            batch = utterances[i:i+BATCH_SIZE]
            audios = []
            for u in batch:
                audio, _ = librosa.load(DATA_DIR / u["audio_path"], sr=SAMPLE_RATE, mono=True)
                audios.append(audio)
            texts = transcribe_batch(model_sm, proc_sm, audios)
            for u, t in zip(batch, texts):
                predictions_b[u["utterance_id"]] = normalizer(t)

        # Merge: prefer large model, fall back to small if large produced empty
        for uid in predictions:
            if len(predictions[uid].strip()) == 0 and len(predictions_b.get(uid, "").strip()) > 0:
                predictions[uid] = predictions_b[uid]

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "submission.jsonl", "w") as f:
        for u in utterances:
            uid = u["utterance_id"]
            line = {"utterance_id": uid, "orthographic_text": predictions.get(uid, "")}
            f.write(json.dumps(line) + "\n")

    print(f"Total time: {time.time() - t0:.0f}s, Predictions: {len(predictions)}")

if __name__ == "__main__":
    main()
```

### 6.2 Silence / Edge Case Handling

```python
import numpy as np

def is_silence(audio, threshold_db=-40):
    """Return True if audio is effectively silent."""
    rms = np.sqrt(np.mean(audio ** 2))
    db = 20 * np.log10(rms + 1e-10)
    return db < threshold_db
```

For silent clips, return `""` (empty string) — the normalizer handles this gracefully and the WER impact is minimal compared to hallucinated text.


## 7. MacBook Development Workflow

This is how you'll spend most of your time locally — writing code, debugging, and testing. GPU training only happens in focused Kaggle sessions.

### 7.1 Project Structure (Local)

```
pasketti/
├── README.md
├── requirements.md
├── design.md
├── requirements.txt
├── src/
│   ├── preprocess.py          # Data loading & preprocessing
│   ├── augment.py             # Noise augmentation pipeline
│   ├── dataset.py             # PyTorch Dataset class
│   ├── train_whisper_small.py # Full fine-tune script
│   ├── train_whisper_lora.py  # LoRA fine-tune script
│   ├── evaluate.py            # WER evaluation with normalizer
│   └── utils.py               # Shared utilities
├── notebooks/
│   ├── 01_eda.ipynb           # Data exploration (MacBook)
│   ├── 02_train_small.ipynb   # Kaggle training notebook
│   ├── 03_train_lora.ipynb    # Kaggle training notebook
│   └── 04_augmented.ipynb     # Kaggle augmented training
├── submission/
│   ├── main.py                # Inference entrypoint
│   ├── model_weights/         # Downloaded from HF Hub before submission
│   └── utils/
├── tests/
│   ├── test_preprocess.py
│   ├── test_inference.py
│   └── test_submission.py
├── data/                      # Git-ignored, local subset
│   ├── audio_sample/          # ~100 FLAC files for testing
│   └── train_word_transcripts.jsonl
└── configs/
    └── training_config.yaml
```

### 7.2 What Runs Where

| Task                           | Where         | Time     | Cost |
|--------------------------------|---------------|----------|------|
| Data exploration & EDA         | MacBook (CPU) | Day 1    | $0   |
| Write preprocessing pipeline   | MacBook       | Day 1–2  | $0   |
| Write Dataset class            | MacBook       | Day 2    | $0   |
| Write training scripts         | MacBook       | Day 2–3  | $0   |
| Test inference on 10 samples   | MacBook (MPS) | Day 3    | $0   |
| Full fine-tune whisper-small   | Kaggle T4     | ~6 hrs   | $0   |
| LoRA fine-tune whisper-large   | Kaggle T4     | ~8 hrs   | $0   |
| Augmented re-training          | Kaggle T4     | ~8 hrs   | $0   |
| Build submission zip           | MacBook       | 1 hr     | $0   |
| Test in Docker runtime         | MacBook       | 1 hr     | $0   |
| Submit to DrivenData           | Browser       | 5 min    | $0   |

### 7.3 MacBook-Specific Tips

**MPS (Metal) backend for quick local tests:**
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Use for inference testing only — MPS training is unreliable for Whisper
```

**Storage management** — a full competition dataset is 10+ GB. Keep only a ~100-file sample locally:
```bash
# Download just 100 random audio files for local development
head -100 train_word_transcripts.jsonl | jq -r '.audio_path' | \
    xargs -I{} cp "full_data/{}" data/audio_sample/
```

**Docker runtime testing** (requires Docker Desktop for Mac):
```bash
git clone https://github.com/drivendataorg/childrens-speech-recognition-runtime.git
cd childrens-speech-recognition-runtime
# Place your submission/ folder contents appropriately
make test-submission
```


## 8. Budget Breakdown

| Item                        | Monthly Cost | Notes                                    |
|-----------------------------|-------------|------------------------------------------|
| Kaggle GPU (30 hrs/wk)     | $0          | Free tier, 2× T4                         |
| Colab GPU (~20 hrs/wk)     | $0          | Free tier, 1× T4                         |
| HuggingFace Hub (private)  | $0          | Free for personal repos                  |
| DrivenData account         | $0          | Free                                     |
| TalkBank account            | $0          | Free with access request                 |
| Docker Desktop (Mac)       | $0          | Free for personal use                    |
| **Subtotal (free path)**   | **$0**      | Fully viable with just free tiers        |
| ─────────────────────       | ─────       | ─────────────────────                    |
| Colab Pro (optional)        | ~$10        | More GPU hours, less disconnections       |
| GCP credits (one-time)     | $0          | $300 free for new accounts               |
| **Maximum monthly spend**  | **$0–10**   | Only if Colab free tier is insufficient  |

**Recommendation**: Start with $0 (Kaggle + Colab free). Only upgrade to Colab Pro if you're consistently running out of GPU hours in Phase 3–4.


## 9. Key Technical Decisions & Rationale

**Why Whisper over Parakeet/NeMo?**
Parakeet TDT 0.6B has better published WER (8.5% vs 9.1% for Whisper-large-v3) but NeMo's adapter training workflow is more complex, less documented for LoRA-style PEFT, and harder to debug on MacBook. Whisper has the deepest ecosystem (HuggingFace, PEFT, bitsandbytes, faster-whisper) and the most community fine-tuning examples. If time permits after Phase 4, Parakeet can be added as a Phase 5 stretch goal.

**Why LoRA instead of full fine-tuning for large-v3?**
Whisper-large-v3 at 1.55B params requires ~24 GB VRAM for full fine-tuning — impossible on T4 (16 GB). LoRA+INT8 fits in ~8 GB. Research shows LoRA matches full fine-tuning for adapter-style domain adaptation, and the frozen base weights prevent catastrophic forgetting on out-of-domain test data.

**Why ensemble instead of single best model?**
Ensembles consistently win ASR competitions. Two models with different error profiles (large-v3 is better on complex utterances; small is faster with less hallucination on short clips) provide complementary coverage. The A100 has enough VRAM and time to run both sequentially.

**Why SpecAugment matters disproportionately?**
Whisper ships with SpecAugment disabled. Enabling it is a single config change that provides free regularization. Multiple practitioners report measurable WER improvements from this alone — it's the highest-ROI change available.

**Why sort by duration at inference?**
Batching utterances of similar length minimizes padding waste. Sorting longest-first ensures the GPU is fully utilized early (when the time budget is fresh) and short utterances zip through at the end.


## 10. Monitoring & Iteration

### Local Validation Dashboard (MacBook)
Track these metrics after each training run:

```
┌─────────────────────────────────────────┐
│         Validation Metrics              │
├─────────────┬───────────┬───────────────┤
│ Metric      │ Clean     │ Noisy (10dB)  │
├─────────────┼───────────┼───────────────┤
│ Overall WER │ 0.XXX     │ 0.XXX         │
│ WER (3-4)   │ 0.XXX     │ 0.XXX         │
│ WER (5-7)   │ 0.XXX     │ 0.XXX         │
│ WER (8-11)  │ 0.XXX     │ 0.XXX         │
│ WER (12+)   │ 0.XXX     │ 0.XXX         │
│ Empty preds │ X / N     │ X / N         │
│ Halluc. %   │ X.X%      │ X.X%          │
└─────────────┴───────────┴───────────────┘
```

**Hallucination detection**: Flag predictions where predicted word count > 3× reference word count. Common with Whisper on very short or silent clips.

### Error Analysis Checklist
After each submission, analyze failures:
1. Which age bucket has highest WER?
2. Are errors concentrated in specific sessions (recording quality)?
3. What fraction of errors are substitutions vs insertions vs deletions?
4. Do noisy samples dominate the error count?
5. Any systematic patterns (e.g., always mis-transcribing "th" sounds)?

Use findings to guide augmentation strategy and potential age-specific adapters.

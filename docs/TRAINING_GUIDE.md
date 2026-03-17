# ChildWhisper Training Guide

Complete step-by-step instructions for data preparation, model training, weight management, and submission packaging.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Data Acquisition](#2-data-acquisition)
3. [Data Access via Google Drive](#3-data-access-via-google-drive)
4. [Data Directory Setup](#4-data-directory-setup)
5. [Train/Validation Split](#5-trainvalidation-split)
6. [Local Dry Run (MacBook)](#6-local-dry-run-macbook)
7. [Training Whisper-small (Full Fine-Tune)](#7-training-whisper-small-full-fine-tune)
8. [Training Whisper-large-v3 (LoRA)](#8-training-whisper-large-v3-lora)
9. [Training with Noise Augmentation](#9-training-with-noise-augmentation)
10. [Saving and Managing Model Weights](#10-saving-and-managing-model-weights)
11. [Downloading Trained Weights](#11-downloading-trained-weights)
12. [Running Evaluation](#12-running-evaluation)
13. [Building the Submission](#13-building-the-submission)
14. [Kaggle Notebook Setup](#14-kaggle-notebook-setup)
15. [Google Colab Notebook Setup](#15-google-colab-notebook-setup)
16. [HuggingFace Hub Token Setup](#16-huggingface-hub-token-setup)
17. [AutoWhisper: Autonomous Experiment Loop](#17-autowhisper-autonomous-experiment-loop)
18. [Troubleshooting](#18-troubleshooting)
19. [Quick Reference: All CLI Commands](#19-quick-reference-all-cli-commands)

---

## 1. Prerequisites

### Software

```bash
# Python 3.11 required
python3 --version  # Should show 3.11.x

# Install all dependencies
pip install -r requirements.txt
```

### Dependency List (requirements.txt)

| Package | Purpose |
|---------|---------|
| torch>=2.1 | Core ML framework |
| torchaudio>=2.1 | Audio I/O |
| transformers>=4.44 | Whisper models |
| peft>=0.12 | LoRA adapters |
| accelerate>=0.33 | Distributed training utils |
| bitsandbytes>=0.43 | INT8 quantization |
| datasets>=2.20 | Dataset utilities |
| librosa>=0.10 | Audio preprocessing |
| soundfile>=0.12 | FLAC I/O |
| jiwer>=3.0 | WER computation |
| audiomentations>=0.36 | Noise augmentation |
| pytest>=7.0 | Testing |
| ruff>=0.4 | Linting |
| pyyaml>=6.0 | Config parsing |

### Accounts Required

| Service | URL | Purpose |
|---------|-----|---------|
| DrivenData | https://www.drivendata.org/competitions/308/childrens-word-asr/ | Competition data |
| HuggingFace | https://huggingface.co | Model checkpoint storage |
| Kaggle | https://www.kaggle.com | Free T4 GPU (30 hrs/week) |
| Google Colab | https://colab.research.google.com | Free T4 GPU (alternative) |

### HuggingFace Login

You must authenticate before training (needed for checkpoint uploads):

```bash
pip install huggingface-hub
huggingface-cli login
# Paste your HuggingFace access token (get it from https://huggingface.co/settings/tokens)
```

---

## 2. Data Acquisition

### Step 2.1: Download Competition Audio

1. Go to https://www.drivendata.org/competitions/308/childrens-word-asr/
2. Register/login and accept the competition rules
3. Navigate to the **Data** tab
4. Download the audio zip files (FLAC format)
5. Download `train_word_transcripts.jsonl` (metadata with utterance IDs, child IDs, age buckets, and transcripts)

### Step 2.2: Request TalkBank Access (Optional)

TalkBank provides additional children's speech data:
1. Go to https://talkbank.org
2. Request access to the relevant corpora
3. Download additional audio if approved

### Step 2.3: Download Noise Datasets (Optional, for Augmentation)

For noise augmentation during training, you need two noise corpora:

**MUSAN (Music, Speech, And Noise):**
```bash
# Download MUSAN noise corpus (~11 GB)
wget https://www.openslr.org/resources/17/musan.tar.gz
tar -xzf musan.tar.gz
# Use the musan/noise/ subdirectory
```

**RealClass (Classroom Noise):**
- Available from academic sources; search for "RealClass classroom noise dataset"
- Contains real classroom background noise recordings

---

## 3. Data Access via Google Drive

Google Drive serves as persistent storage between ephemeral Kaggle/Colab sessions.

### Step 3.1: Upload Data to Google Drive

On your local machine, upload competition data to Google Drive:

```
Google Drive/
└── childwhisper/
    └── data/
        ├── audio/                          # All FLAC audio files
        ├── train_word_transcripts.jsonl     # Competition metadata
        ├── musan_noise/                     # (Optional) MUSAN noise files
        └── realclass_noise/                 # (Optional) RealClass noise files
```

### Step 3.2: Access from Google Colab

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Copy data to fast local storage (much faster I/O than Drive)
!cp -r /content/drive/MyDrive/childwhisper/data /content/data

# Verify
!ls /content/data/
# Should show: audio/  train_word_transcripts.jsonl  (and optionally noise dirs)

!wc -l /content/data/train_word_transcripts.jsonl
# Shows total number of utterances
```

### Step 3.3: Access from Kaggle

**Option A: Kaggle Dataset Upload**
1. Go to https://www.kaggle.com/datasets/create
2. Upload the audio files and metadata as a new private dataset
3. In your notebook, add the dataset via "Add Data" sidebar
4. Access at `/kaggle/input/your-dataset-name/`

**Option B: Google Drive via Kaggle**
1. In Kaggle notebook settings, enable "Google Drive" under Add-ons
2. Mount and copy:

```python
from google.colab import drive  # Kaggle supports this
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/childwhisper/data /kaggle/working/data
```

### Step 3.4: Using gdown for Shared Drive Links

If you share specific files via Google Drive public links:

```bash
pip install gdown

# Download a single file by ID
gdown --id <GOOGLE_DRIVE_FILE_ID> -O data/audio.zip

# Download an entire folder
gdown --folder <GOOGLE_DRIVE_FOLDER_URL> -O data/

# Extract
unzip data/audio.zip -d data/audio/
```

To get the file ID: right-click the file in Google Drive > "Get link" > the ID is the long string in the URL between `/d/` and `/view`.

---

## 4. Data Directory Setup

After downloading, your data directory must look like this:

```
data/
├── audio/                              # ALL FLAC files from competition
│   ├── utterance_00001.flac
│   ├── utterance_00002.flac
│   └── ... (thousands of files)
├── train_word_transcripts.jsonl        # Metadata file
├── audio_sample/                       # ~100 files for local MacBook testing
│   ├── utterance_00001.flac
│   ├── utterance_00042.flac
│   └── ... (~100 random files)
├── musan_noise/                        # (Optional) MUSAN noise audio
│   ├── noise-free-sound-0001.wav
│   └── ...
└── realclass_noise/                    # (Optional) RealClass noise audio
    ├── classroom_001.wav
    └── ...
```

### Create the Local Test Sample

Copy ~100 random FLAC files for local MacBook testing:

```bash
mkdir -p data/audio_sample

# Copy 100 random files
ls data/audio/*.flac | shuf | head -100 | xargs -I {} cp {} data/audio_sample/
```

### Metadata Format

Each line in `train_word_transcripts.jsonl` is a JSON object:

```json
{"utterance_id": "utt_00001", "child_id": "child_042", "age_bucket": "4-5", "audio_path": "utterance_00001.flac", "transcript": "hello"}
```

Fields used by the training pipeline:
- `utterance_id` — unique identifier
- `child_id` — used for train/val splitting (no speaker leakage)
- `age_bucket` — used for stratified splitting and per-age WER tracking
- `audio_path` — filename of the FLAC file (relative to audio_dir)
- `transcript` — ground truth transcription

---

## 5. Train/Validation Split

The split is **automatic** — the training scripts handle it internally. Here's how it works:

### How It Works

- **Split by `child_id`**: All utterances from one child go to the SAME split (prevents speaker leakage)
- **Stratified by `age_bucket`**: Each age group is proportionally represented in both splits
- **90/10 ratio**: 90% training, 10% validation
- Configured in `configs/training_config.yaml`:

```yaml
common:
  validation:
    split_ratio: 0.1          # 10% for validation
    stratify_by: age_bucket   # Stratify by age group
    split_by: child_id        # Split by speaker, not utterance
```

### Verify the Split Manually (Optional)

```python
from src.dataset import create_train_val_split
from src.preprocess import load_metadata

metadata = load_metadata("data/train_word_transcripts.jsonl")
train_meta, val_meta = create_train_val_split(
    metadata,
    val_ratio=0.1,
    split_by="child_id",
    stratify_by="age_bucket",
)

print(f"Total utterances: {len(metadata)}")
print(f"Train utterances: {len(train_meta)}")
print(f"Val utterances:   {len(val_meta)}")

# Check no speaker leakage
train_children = {e["child_id"] for e in train_meta}
val_children = {e["child_id"] for e in val_meta}
overlap = train_children & val_children
print(f"Speaker overlap:  {len(overlap)} (should be 0)")
```

### Test Data

Test data has no transcripts — it's what you run inference on for submission. The test metadata file (`utterance_metadata.jsonl`) is provided in the competition runtime environment at `/code_execution/data/`.

---

## 6. Local Dry Run (MacBook)

**Always test locally before using GPU hours.** Dry run mode runs 1 training step + 1 eval step on CPU/MPS.

### Step 6.1: Dry Run Whisper-small

```bash
python src/train_whisper_small.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio_sample \
  --config configs/training_config.yaml \
  --output-dir output/whisper-small-dry \
  --dry-run \
  --no-push-to-hub
```

What this does:
- Loads the config and merges `common` + `whisper_small` sections
- Creates train/val split from the metadata
- Loads Whisper-small model (~967 MB download on first run)
- Runs 1 training step on a mini batch
- Runs 1 evaluation step
- Skips FP16, gradient checkpointing, and Hub push
- Verifies the full pipeline works end-to-end

### Step 6.2: Dry Run Whisper-large-v3 LoRA

```bash
python src/train_whisper_lora.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio_sample \
  --config configs/training_config.yaml \
  --output-dir output/whisper-lora-dry \
  --dry-run \
  --no-push-to-hub
```

What this does:
- Same as above but loads Whisper-large-v3 (~6.2 GB download on first run)
- Applies LoRA adapter (r=32, alpha=64 on q_proj + v_proj)
- Skips INT8 quantization in dry-run mode (bitsandbytes may not work on macOS)
- Verifies LoRA injection and training loop

### Expected Output

```
INFO - Loading config from configs/training_config.yaml
INFO - Dry run mode: 1 train step, 1 eval step
INFO - Loading metadata from data/train_word_transcripts.jsonl
INFO - Train/val split: 90 train, 10 val
INFO - Model loaded: openai/whisper-small
INFO - Training...
INFO - Step 1/1 complete
INFO - Evaluating...
INFO - Validation WER: 0.XXXX
INFO - Done.
```

If you see errors, fix them before proceeding to GPU training.

---

## 7. Training Whisper-small (Full Fine-Tune)

### Training Environment

- **Where**: Kaggle T4 (16 GB VRAM) or Google Colab T4
- **Time**: ~2-3 hours for 3 epochs on full dataset
- **VRAM usage**: ~12-14 GB (FP16 + gradient checkpointing)

### CLI Arguments Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--metadata-path` | Yes | — | Path to `train_word_transcripts.jsonl` |
| `--audio-dir` | Yes | — | Directory containing FLAC audio files |
| `--config` | No | `configs/training_config.yaml` | Training config YAML |
| `--output-dir` | No | `./checkpoints/whisper-small` | Where to save checkpoints |
| `--num-train-epochs` | No | 3 (from config) | Override epoch count |
| `--dry-run` | No | false | Run 1 step only |
| `--no-push-to-hub` | No | false | Disable HF Hub upload |
| `--noise-dir` | No | None | MUSAN noise directory |
| `--realclass-dir` | No | None | RealClass noise directory |
| `--hub-model-id` | No | from config | Override Hub repo name |

### Step 7.1: Basic Training (No Augmentation)

```bash
python src/train_whisper_small.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --config configs/training_config.yaml \
  --output-dir ./checkpoints/whisper-small
```

### Step 7.2: Training with Custom Hub ID

```bash
python src/train_whisper_small.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --config configs/training_config.yaml \
  --output-dir ./checkpoints/whisper-small \
  --hub-model-id nishantgaurav23/pasketti-whisper-small
```

### Step 7.3: Training without Hub Push (Save Locally Only)

```bash
python src/train_whisper_small.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --config configs/training_config.yaml \
  --output-dir ./checkpoints/whisper-small \
  --no-push-to-hub
```

### Training Config Details (whisper_small section)

```yaml
whisper_small:
  model_name: openai/whisper-small        # 244M parameters
  learning_rate: 1.0e-5                    # Conservative LR for full FT
  warmup_steps: 500                        # Linear warmup
  num_train_epochs: 3                      # 3 full passes
  per_device_train_batch_size: 2           # Fits on T4
  per_device_eval_batch_size: 4            # Larger eval batch (no gradients)
  gradient_accumulation_steps: 8           # Effective batch = 2 * 8 = 16
  fp16: true                               # Half precision
  gradient_checkpointing: true             # Trade compute for memory
  eval_steps: 500                          # Evaluate every 500 steps
  save_steps: 500                          # Checkpoint every 500 steps
  save_total_limit: 3                      # Keep 3 best checkpoints
  generation_max_length: 225               # Max tokens per prediction
  hub_model_id: nishantgaurav23/pasketti-whisper-small
  hub_private_repo: true                   # Private HF repo
```

### What Happens During Training

1. Config loaded from YAML (merges `common` + `whisper_small`)
2. Metadata loaded from JSONL
3. Train/val split created by `child_id` with `age_bucket` stratification
4. Whisper-small model loaded from HuggingFace Hub
5. Gradient checkpointing enabled (saves VRAM)
6. SpecAugment enabled (mask_time_prob=0.05, mask_feature_prob=0.04)
7. `Seq2SeqTrainer` runs training loop:
   - Each step: load batch → extract Mel features → forward pass → compute loss → backward → update
   - Every 500 steps: run evaluation on val set → compute WER → save checkpoint
   - Best checkpoint determined by lowest WER
8. After all epochs: final evaluation, save model + processor, push to Hub

### Expected Training Output

```
Epoch 1/3: 100%|████████| 1250/1250 [45:00<00:00]
  Training loss: 0.45
  Validation WER: 0.32

Epoch 2/3: 100%|████████| 1250/1250 [45:00<00:00]
  Training loss: 0.28
  Validation WER: 0.24

Epoch 3/3: 100%|████████| 1250/1250 [45:00<00:00]
  Training loss: 0.19
  Validation WER: 0.21

Final Validation WER: 0.21
Model saved to ./checkpoints/whisper-small
Pushed to nishantgaurav23/pasketti-whisper-small
```

---

## 8. Training Whisper-large-v3 (LoRA)

### Training Environment

- **Where**: Kaggle T4 (16 GB VRAM) or Google Colab T4
- **Time**: ~4-5 hours for 3 epochs on full dataset
- **VRAM usage**: ~10-12 GB (INT8 base + LoRA adapter + FP16 gradients)

### CLI Arguments Reference

Same as Whisper-small (see table above), plus INT8 and LoRA handled automatically from config.

### Step 8.1: Basic Training

```bash
python src/train_whisper_lora.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --config configs/training_config.yaml \
  --output-dir ./checkpoints/whisper-large-v3-lora
```

### Step 8.2: Training with Custom Hub ID

```bash
python src/train_whisper_lora.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --config configs/training_config.yaml \
  --output-dir ./checkpoints/whisper-large-v3-lora \
  --hub-model-id nishantgaurav23/pasketti-whisper-lora
```

### Training Config Details (whisper_large_v3 section)

```yaml
whisper_large_v3:
  model_name: openai/whisper-large-v3      # 1.55B parameters
  learning_rate: 1.0e-3                     # Higher LR — only LoRA weights update
  warmup_steps: 500
  num_train_epochs: 3
  per_device_train_batch_size: 1            # INT8 + LoRA limits batch to 1
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 16           # Effective batch = 1 * 16 = 16
  fp16: true
  gradient_checkpointing: true
  load_in_8bit: true                        # INT8 quantization via bitsandbytes
  eval_steps: 500
  save_steps: 500
  save_total_limit: 3
  generation_max_length: 225
  hub_model_id: nishantgaurav23/pasketti-whisper-lora
  hub_private_repo: true

  lora:
    r: 32                                   # LoRA rank
    alpha: 64                               # Scaling factor (alpha/r = 2)
    target_modules:                         # Which layers get LoRA
      - q_proj                              # Attention query projection
      - v_proj                              # Attention value projection
    dropout: 0.05                           # LoRA dropout
    bias: none                              # Don't train bias terms
    task_type: SEQ_2_SEQ_LM                 # Encoder-decoder task
```

### What Happens During Training

1. Config loaded (merges `common` + `whisper_large_v3`)
2. Whisper-large-v3 loaded with INT8 quantization:
   - `load_in_8bit=True` reduces 1.55B params from ~6.2 GB to ~1.6 GB
   - `device_map="auto"` handles layer placement
3. `prepare_model_for_kbit_training()` freezes non-trainable base weights
4. LoRA adapter injected:
   - Adds small trainable matrices to `q_proj` and `v_proj` in every attention layer
   - Only ~15M trainable parameters (1% of total)
   - Total trainable: ~63 MB vs ~6.2 GB for full model
5. Training loop runs same as Whisper-small
6. **Only the LoRA adapter is saved** (not the full 1.55B base model)

### Why LoRA + INT8?

| | Full Fine-Tune | LoRA + INT8 |
|---|---|---|
| Trainable params | 1.55B | ~15M |
| Base model VRAM | ~6.2 GB (FP16) | ~1.6 GB (INT8) |
| Gradient VRAM | ~6.2 GB | ~60 MB |
| Total VRAM | >16 GB (OOM on T4) | ~10-12 GB (fits T4) |
| Checkpoint size | ~6.2 GB | ~63 MB |
| Performance | Best | ~95-99% of full FT |

---

## 9. Training with Noise Augmentation

Noise augmentation improves robustness for noisy classroom recordings. **Both `--noise-dir` and `--realclass-dir` must be provided together** (the script raises an error if only one is given).

### Augmentation Pipeline

When enabled, each training sample is randomly augmented:
- **50% chance**: RealClass classroom noise added (SNR 5-20 dB)
- **20% chance**: MUSAN babble noise added (SNR 0-15 dB)
- **30% chance**: Clean (no noise added)
- **All samples**: Optional gain variation of +/-6 dB (30% probability)

Validation data is **never** augmented.

### Step 9.1: Whisper-small with Augmentation

```bash
python src/train_whisper_small.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --config configs/training_config.yaml \
  --output-dir ./checkpoints/whisper-small-augmented \
  --noise-dir data/musan_noise \
  --realclass-dir data/realclass_noise
```

### Step 9.2: Whisper-large-v3 LoRA with Augmentation

```bash
python src/train_whisper_lora.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --config configs/training_config.yaml \
  --output-dir ./checkpoints/whisper-lora-augmented \
  --noise-dir data/musan_noise \
  --realclass-dir data/realclass_noise
```

### Augmentation Config (from training_config.yaml)

```yaml
common:
  augmentation:
    realclass_min_snr_db: 5.0     # Classroom noise: 5 to 20 dB SNR
    realclass_max_snr_db: 20.0
    musan_min_snr_db: 0.0         # Babble noise: 0 to 15 dB SNR
    musan_max_snr_db: 15.0
```

---

## 10. Saving and Managing Model Weights

### Automatic Saving (During Training)

Both training scripts save checkpoints automatically:

1. **Local checkpoints**: Saved to `--output-dir` every 500 steps
   - Keeps the 3 best checkpoints (by validation WER)
   - Older checkpoints are deleted automatically

2. **HuggingFace Hub**: Pushed every 500 steps (unless `--no-push-to-hub`)
   - Whisper-small → `nishantgaurav23/pasketti-whisper-small` (~500 MB)
   - Whisper-large-v3 LoRA → `nishantgaurav23/pasketti-whisper-lora` (~63 MB)
   - Both as private repos

### What Gets Saved

**Whisper-small (full fine-tune):**
```
checkpoints/whisper-small/
├── config.json                    # Model architecture config
├── generation_config.json         # Generation parameters
├── model.safetensors              # Full model weights (~500 MB)
├── preprocessor_config.json       # Feature extractor config
├── special_tokens_map.json        # Tokenizer special tokens
├── tokenizer.json                 # Tokenizer vocabulary
└── vocab.json                     # Vocabulary mapping
```

**Whisper-large-v3 LoRA (adapter only):**
```
checkpoints/whisper-large-v3-lora/
├── adapter_config.json            # LoRA configuration
├── adapter_model.safetensors      # LoRA weights only (~63 MB)
├── config.json                    # Base model config reference
├── generation_config.json
├── preprocessor_config.json
├── special_tokens_map.json
└── tokenizer.json
```

### Why Hub Checkpointing Matters

Kaggle and Colab sessions are **ephemeral** — your VM can be killed at any time. Pushing to HuggingFace Hub every 500 steps ensures you never lose more than 500 steps of training progress.

### Manual Upload to HuggingFace Hub

If you trained with `--no-push-to-hub` and want to upload later:

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload Whisper-small
api.upload_folder(
    folder_path="./checkpoints/whisper-small",
    repo_id="nishantgaurav23/pasketti-whisper-small",
    repo_type="model",
    private=True,
)

# Upload LoRA adapter
api.upload_folder(
    folder_path="./checkpoints/whisper-large-v3-lora",
    repo_id="nishantgaurav23/pasketti-whisper-lora",
    repo_type="model",
    private=True,
)
```

### Copy Weights to Google Drive (Backup)

```python
# In Colab/Kaggle — copy checkpoints to Drive as backup
!cp -r ./checkpoints/whisper-small /content/drive/MyDrive/childwhisper/checkpoints/
!cp -r ./checkpoints/whisper-large-v3-lora /content/drive/MyDrive/childwhisper/checkpoints/
```

---

## 11. Downloading Trained Weights

After training is complete, download the weights for local testing and submission packaging.

### Step 11.1: Install HuggingFace CLI

```bash
pip install huggingface-hub
huggingface-cli login
```

### Step 11.2: Download LoRA Adapter

```bash
huggingface-cli download nishantgaurav23/pasketti-whisper-lora \
  --local-dir submission/model_weights/lora_large_v3
```

### Step 11.3: Download Fine-Tuned Whisper-small

```bash
huggingface-cli download nishantgaurav23/pasketti-whisper-small \
  --local-dir submission/model_weights/whisper_small_ft
```

### Step 11.4: Verify Download

```bash
ls -la submission/model_weights/lora_large_v3/
# Should show adapter_config.json, adapter_model.safetensors, etc.

ls -la submission/model_weights/whisper_small_ft/
# Should show config.json, model.safetensors (~500 MB), etc.
```

### Expected Directory After Download

```
submission/
├── main.py
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   └── utils.py
└── model_weights/
    ├── lora_large_v3/          # ~63 MB
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── whisper_small_ft/       # ~500 MB
        ├── config.json
        ├── model.safetensors
        ├── preprocessor_config.json
        └── tokenizer.json
```

---

## 12. Running Evaluation

### Step 12.1: Local Inference Test

Test the inference pipeline on your MacBook with the sample data:

```python
# In Python or a notebook
import json
from pathlib import Path
from src.preprocess import load_metadata, load_audio, trim_silence, is_silence
from src.utils import normalize_and_postprocess
from src.evaluate import compute_wer, validation_summary, error_analysis_summary

# Load metadata
metadata = load_metadata("data/train_word_transcripts.jsonl")

# The submission/main.py can be tested directly:
# It auto-detects device (cuda > mps > cpu)
```

Or run the full inference pipeline:

```bash
cd submission
python main.py
```

The inference pipeline (`submission/main.py`):
1. Detects device (CUDA > MPS > CPU)
2. Loads Model A (Whisper-large-v3 + LoRA adapter)
3. Sorts utterances by duration (longest first for efficient batching)
4. Runs batch inference with beam search (num_beams=8 for large)
5. If time allows (<90 min elapsed), loads Model B (Whisper-small fine-tuned)
6. Merges predictions: prefers Model A, falls back to Model B on empty predictions
7. Applies `normalize_and_postprocess()` to all predictions
8. Writes `submission.jsonl`

### Step 12.2: Compute WER on Validation Set

```python
from src.evaluate import (
    compute_wer,
    compute_per_age_wer,
    validation_summary,
    format_validation_report,
    error_analysis_summary,
    format_error_analysis_report,
)

# After running inference and collecting predictions
references = [entry["transcript"] for entry in val_meta]
hypotheses = [predictions[entry["utterance_id"]] for entry in val_meta]
age_buckets = [entry["age_bucket"] for entry in val_meta]

# Overall WER
wer = compute_wer(references, hypotheses)
print(f"Overall WER: {wer:.4f}")

# Per-age-bucket WER
per_age = compute_per_age_wer(references, hypotheses, age_buckets)
for bucket, bucket_wer in sorted(per_age.items()):
    print(f"  {bucket}: {bucket_wer:.4f}")

# Full validation summary
summary = validation_summary(references, hypotheses, age_buckets)

# Detailed error analysis (substitutions, insertions, deletions, hallucinations)
analysis = error_analysis_summary(references, hypotheses, age_buckets)
print(format_error_analysis_report(analysis))
```

### Step 12.3: Noisy Validation

Test robustness with synthetic classroom noise:

```python
from src.evaluate import apply_noise_to_val, combined_validation_summary

# Load validation audio
val_audio = [load_audio(f"data/audio/{e['audio_path']}") for e in val_meta]

# Add noise at 10 dB SNR
noisy_audio = apply_noise_to_val(val_audio, noise_dir="data/realclass_noise", snr_db=10.0)

# Run inference on noisy audio and compute combined summary
combined = combined_validation_summary(references, clean_hyps, noisy_hyps, age_buckets)
print(f"Clean WER:  {combined['clean']['overall_wer']:.4f}")
print(f"Noisy WER:  {combined['noisy']['overall_wer']:.4f}")
print(f"Degradation: {combined['wer_degradation']:.4f}")
```

---

## 13. Building the Submission

### Step 13.1: Ensure Weights Are Downloaded

```bash
# Verify model weights exist
ls submission/model_weights/lora_large_v3/adapter_model.safetensors
ls submission/model_weights/whisper_small_ft/model.safetensors
```

### Step 13.2: Dry Run (Validate Without Building)

```bash
bash scripts/build_submission.sh --dry-run
```

This will:
- Copy `src/preprocess.py` and `src/utils.py` into `submission/src/`
- Validate the submission directory structure
- Report file sizes and total budget
- List all files that would be included
- **NOT** create the zip file

### Step 13.3: Build submission.zip

```bash
bash scripts/build_submission.sh
```

Output:
```
=== ChildWhisper Submission Builder ===
Bundling src/ into submission/...
  Copied preprocess.py, utils.py
Validating submission directory...
  Structure OK

Size budget:
  Code:          45,000 bytes
  Weights:  563,000,000 bytes
  Total:   ~537 MB

Files: 12

Building zip...
Created: submission.zip (536,234,567 bytes)

Upload at: https://www.drivendata.org/competitions/308/childrens-word-asr/
Done.
```

### Step 13.4: Custom Output Path

```bash
bash scripts/build_submission.sh --output /path/to/my_submission.zip
```

### Step 13.5: Upload to DrivenData

1. Go to https://www.drivendata.org/competitions/308/childrens-word-asr/
2. Navigate to the **Submit** tab
3. Upload `submission.zip`
4. Wait for the runtime to process (runs on A100 80GB, 2-hour limit, no internet)

### What's in the Submission

```
submission.zip
├── main.py                              # Inference entrypoint (called by runtime)
├── src/
│   ├── __init__.py
│   ├── preprocess.py                    # Audio loading, silence detection
│   └── utils.py                         # Text normalization, post-processing
└── model_weights/
    ├── lora_large_v3/                   # ~63 MB (LoRA adapter only)
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── whisper_small_ft/                # ~500 MB (full fine-tuned model)
        ├── config.json
        ├── model.safetensors
        ├── preprocessor_config.json
        └── tokenizer.json
```

The A100 runtime has `openai/whisper-large-v3` pre-installed, so the LoRA adapter is loaded on top of it. The Whisper-small weights must be fully bundled since it's fine-tuned.

---

## 14. Kaggle Notebook Setup

Complete notebook template for training on Kaggle:

```python
# ============================================================
# Cell 1: Install Dependencies
# ============================================================
!pip install -q transformers peft accelerate bitsandbytes \
    librosa soundfile jiwer audiomentations torchaudio pyyaml tqdm

# ============================================================
# Cell 2: Clone Repository
# ============================================================
!git clone https://github.com/YOUR_USERNAME/childwhisper.git
%cd childwhisper

# ============================================================
# Cell 3: Setup Data (from Kaggle Dataset)
# ============================================================
# Option A: If data is a Kaggle dataset
!ln -s /kaggle/input/pasketti-data/audio data/audio
!ln -s /kaggle/input/pasketti-data/train_word_transcripts.jsonl \
    data/train_word_transcripts.jsonl

# Option B: If data is on Google Drive
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p data
!cp -r /content/drive/MyDrive/childwhisper/data/audio data/
!cp /content/drive/MyDrive/childwhisper/data/train_word_transcripts.jsonl data/

# ============================================================
# Cell 4: Login to HuggingFace Hub
# ============================================================
from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")  # Replace with your token

# ============================================================
# Cell 5: Verify GPU
# ============================================================
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================
# Cell 6: Train Whisper-small (Full Fine-Tune)
# ============================================================
!python src/train_whisper_small.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --config configs/training_config.yaml \
  --output-dir /kaggle/working/checkpoints/whisper-small \
  --hub-model-id nishantgaurav23/pasketti-whisper-small

# ============================================================
# Cell 7: Train Whisper-large-v3 (LoRA) — run in SEPARATE session
# ============================================================
# NOTE: Run this in a new Kaggle session to get fresh GPU hours.
# The large model + LoRA takes ~4-5 hours.
!python src/train_whisper_lora.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --config configs/training_config.yaml \
  --output-dir /kaggle/working/checkpoints/whisper-lora \
  --hub-model-id nishantgaurav23/pasketti-whisper-lora

# ============================================================
# Cell 8: (Optional) Backup to Google Drive
# ============================================================
!cp -r /kaggle/working/checkpoints /content/drive/MyDrive/childwhisper/
```

### Kaggle Tips

- **GPU quota**: 30 hours/week of T4 GPU time
- **Session limit**: Each session runs for up to 12 hours
- **Train models in separate sessions**: Whisper-small (~3 hrs) and LoRA (~5 hrs)
- **Data persistence**: Kaggle working directory is wiped between sessions — always push to Hub
- **Internet**: Enable internet in notebook settings (needed for Hub push and model downloads)

---

## 15. Google Colab Notebook Setup

Complete notebook template for training on Colab:

```python
# ============================================================
# Cell 1: Check GPU
# ============================================================
!nvidia-smi
# Verify you have a T4 (or better)
# If no GPU: Runtime → Change runtime type → T4 GPU

# ============================================================
# Cell 2: Install Dependencies
# ============================================================
!pip install -q transformers peft accelerate bitsandbytes \
    librosa soundfile jiwer audiomentations torchaudio pyyaml tqdm

# ============================================================
# Cell 3: Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# Cell 4: Copy Data to Local Storage
# ============================================================
# IMPORTANT: Copy to /content/ (local SSD) for fast I/O.
# Training directly from Drive is 10-50x slower.
!mkdir -p /content/data
!cp -r /content/drive/MyDrive/childwhisper/data/audio /content/data/
!cp /content/drive/MyDrive/childwhisper/data/train_word_transcripts.jsonl /content/data/

# Optional: Copy noise data for augmentation
# !cp -r /content/drive/MyDrive/childwhisper/data/musan_noise /content/data/
# !cp -r /content/drive/MyDrive/childwhisper/data/realclass_noise /content/data/

# Verify
!echo "Audio files:" && ls /content/data/audio/ | wc -l
!echo "Metadata lines:" && wc -l /content/data/train_word_transcripts.jsonl

# ============================================================
# Cell 5: Clone Repository
# ============================================================
!git clone https://github.com/YOUR_USERNAME/childwhisper.git
%cd childwhisper

# ============================================================
# Cell 6: Login to HuggingFace Hub
# ============================================================
from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")  # Replace with your token

# ============================================================
# Cell 7: Verify Setup with Dry Run
# ============================================================
!python src/train_whisper_small.py \
  --metadata-path /content/data/train_word_transcripts.jsonl \
  --audio-dir /content/data/audio \
  --config configs/training_config.yaml \
  --output-dir /content/output/dry \
  --dry-run \
  --no-push-to-hub

# ============================================================
# Cell 8: Train Whisper-small
# ============================================================
!python src/train_whisper_small.py \
  --metadata-path /content/data/train_word_transcripts.jsonl \
  --audio-dir /content/data/audio \
  --config configs/training_config.yaml \
  --output-dir /content/output/whisper-small \
  --hub-model-id nishantgaurav23/pasketti-whisper-small

# ============================================================
# Cell 9: Backup to Drive (after training completes)
# ============================================================
!cp -r /content/output/whisper-small \
  /content/drive/MyDrive/childwhisper/checkpoints/whisper-small

# ============================================================
# Cell 10: Train Whisper-large-v3 LoRA
# ============================================================
# Can run in same session if time permits, or start a new one
!python src/train_whisper_lora.py \
  --metadata-path /content/data/train_word_transcripts.jsonl \
  --audio-dir /content/data/audio \
  --config configs/training_config.yaml \
  --output-dir /content/output/whisper-lora \
  --hub-model-id nishantgaurav23/pasketti-whisper-lora

# ============================================================
# Cell 11: Backup LoRA to Drive
# ============================================================
!cp -r /content/output/whisper-lora \
  /content/drive/MyDrive/childwhisper/checkpoints/whisper-lora
```

### Colab Tips

- **Session limit**: Free tier disconnects after ~90 minutes of inactivity or ~12 hours total
- **Keep alive**: Keep the browser tab active; use a keep-alive extension if needed
- **Local > Drive**: Always copy data from Drive to `/content/` before training — Drive I/O is extremely slow
- **Colab Pro**: If you need more reliable sessions or A100 GPUs, consider Colab Pro ($10/month)

---

## 16. HuggingFace Hub Token Setup

HuggingFace Hub is used to store model checkpoints (both during training and for AutoWhisper). You need a **write-access token** to push models.

### Step 16.1: Create a HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Name: `childwhisper` (or anything descriptive)
4. Role: **Write** (required for pushing models)
5. Click **Generate**
6. Copy the token (starts with `hf_...`)

### Step 16.2: Local Setup (MacBook)

Two options — use whichever you prefer:

**Option A: CLI login (recommended — persists across sessions)**

```bash
# One-time setup. Saves token to ~/.cache/huggingface/token
huggingface-cli login
# Paste your token when prompted
```

Verify it works:

```bash
huggingface-cli whoami
# Should print: nishantgaurav23
```

**Option B: Environment variable (per-session)**

```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc) for persistence
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxx

# Or use a .env file (already .gitignored)
echo "HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxx" >> .env
```

### Step 16.3: Kaggle Setup

1. Open your Kaggle notebook
2. Click **Add-ons** (sidebar) → **Secrets**
3. Click **+ Add a new secret**
4. Label: `HUGGING_FACE_HUB_TOKEN`
5. Value: paste your `hf_...` token
6. Toggle the secret **ON** for the notebook

In the notebook, access it like this:

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

secrets = UserSecretsClient()
hf_token = secrets.get_secret("HUGGING_FACE_HUB_TOKEN")
login(token=hf_token)
```

All existing notebooks (02, 03, 04) and AutoWhisper (05) use this same pattern.

### Step 16.4: Google Colab Setup

Two options:

**Option A: Colab Secrets (recommended)**

1. Click the **key icon** (🔑) in the left sidebar
2. Click **+ Add new secret**
3. Name: `HF_TOKEN`
4. Value: paste your `hf_...` token
5. Toggle notebook access ON

```python
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get("HF_TOKEN"))
```

**Option B: Inline (quick and dirty — avoid for shared notebooks)**

```python
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxxx")  # Don't commit this!
```

### Step 16.5: Verify Token Works

Run this in any environment to confirm:

```python
from huggingface_hub import HfApi

api = HfApi()
user = api.whoami()
print(f"Logged in as: {user['name']}")

# Test repo creation (will be used by AutoWhisper)
api.create_repo(
    repo_id="nishantgaurav23/pasketti-test-delete-me",
    private=True,
    exist_ok=True,
)
# Clean up
api.delete_repo(repo_id="nishantgaurav23/pasketti-test-delete-me")
print("Token has write access — all good!")
```

### HuggingFace Hub Repos Used by ChildWhisper

| Repo ID | Purpose | Created By |
|---------|---------|------------|
| `nishantgaurav23/pasketti-whisper-small` | Fine-tuned Whisper-small checkpoints | S2.2 training |
| `nishantgaurav23/pasketti-whisper-lora` | LoRA adapter for Whisper-large-v3 | S3.1 training |
| `nishantgaurav23/pasketti-whisper-small-augmented` | Augmented Whisper-small | S4.3 training |
| `nishantgaurav23/pasketti-whisper-lora-augmented` | Augmented LoRA adapter | S4.3 training |
| `nishantgaurav23/pasketti-autowhisper-best` | AutoWhisper best experiment result | S6.1 AutoWhisper |

All repos are **auto-created** on first push — you don't need to create them manually on the HF website.

---

## 17. AutoWhisper: Autonomous Experiment Loop

AutoWhisper is an AI-driven experiment loop (inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)) that autonomously modifies a training script, runs time-boxed experiments, keeps improvements, and reverts regressions.

### How It Works

```
┌──────────────────────────────────────────────────┐
│  1. Agent reads program.md + current train.py     │
│  2. Agent proposes a modification to train.py     │
│  3. Git commit the change                         │
│  4. Run: python train.py (budget: 15 min)         │
│  5. Evaluate: parse val_wer from stdout            │
│  6. If WER improved → KEEP (commit stays)         │
│     If WER worsened → REVERT (git reset)          │
│  7. Log result to results.tsv                     │
│  8. Repeat                                        │
└──────────────────────────────────────────────────┘
```

### Step 17.1: Initialize a Run

```bash
# Creates branch autowhisper/run_mar17, runs baseline experiment
python -m src.autowhisper.runner init --tag run_mar17
```

### Step 17.2: Run on Kaggle (Scripted Mode — No API Key Needed)

Scripted mode applies a pre-defined sequence of modifications (no Claude API cost):

```python
# In notebook 05_autowhisper.ipynb

# Cell 1: Setup
!pip install -q transformers peft accelerate bitsandbytes \
    librosa soundfile jiwer audiomentations torchaudio pyyaml tqdm

# Cell 2: Clone and setup
!git clone https://github.com/YOUR_USERNAME/childwhisper.git
%cd childwhisper

# Cell 3: HuggingFace login (using Kaggle secret)
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
secrets = UserSecretsClient()
login(token=secrets.get_secret("HUGGING_FACE_HUB_TOKEN"))

# Cell 4: Data setup
!ln -s /kaggle/input/pasketti-data/audio data/audio
!ln -s /kaggle/input/pasketti-data/train_word_transcripts.jsonl \
    data/train_word_transcripts.jsonl

# Cell 5: Initialize AutoWhisper run
!python -m src.autowhisper.runner init --tag run_mar17

# Cell 6: Run experiment loop (scripted mode)
# This runs pre-defined patches one by one, ~15 min each
!python -m src.autowhisper.runner loop \
    --tag run_mar17 \
    --mode scripted \
    --max-experiments 30 \
    --time-budget 900
```

### Step 17.3: Run on Kaggle (Agent Mode — Requires Claude API Key)

Agent mode lets Claude autonomously propose modifications:

```python
# Additional Kaggle secret needed: ANTHROPIC_API_KEY

# Cell: Add Claude API key
import os
secrets = UserSecretsClient()
os.environ["ANTHROPIC_API_KEY"] = secrets.get_secret("ANTHROPIC_API_KEY")

# Cell: Run in agent mode
!python -m src.autowhisper.runner loop \
    --tag run_mar17 \
    --mode agent \
    --max-experiments 30 \
    --time-budget 900
```

### Step 17.4: HF Hub Checkpointing Strategy

During the loop, **individual experiments do NOT push to HF Hub** (most get reverted). Instead:

1. **During the loop**: All checkpoints are saved locally only (`/kaggle/working/autowhisper/`)
2. **At the end**: Only the **best model** (lowest val_wer) is pushed to HF Hub
3. **The experiment log** (`results.tsv`) is also uploaded as an artifact

The notebook handles this automatically:

```python
# This runs at the end of the loop (or when session time is running low)
from huggingface_hub import HfApi

api = HfApi()

# Auto-creates repo if it doesn't exist
api.create_repo(
    repo_id="nishantgaurav23/pasketti-autowhisper-best",
    private=True,
    exist_ok=True,  # no-op if already exists
)

# Push the best checkpoint
api.upload_folder(
    folder_path="/kaggle/working/autowhisper/best_checkpoint",
    repo_id="nishantgaurav23/pasketti-autowhisper-best",
    repo_type="model",
)

# Push experiment log
api.upload_file(
    path_or_fileobj="results/autowhisper/run_mar17/results.tsv",
    path_in_repo="logs/results_run_mar17.tsv",
    repo_id="nishantgaurav23/pasketti-autowhisper-best",
)
```

### Step 17.5: Session Safety

The notebook monitors Kaggle session time and stops the loop with a 30-minute buffer:

```python
import time

SESSION_START = time.time()
MAX_SESSION_HOURS = 11.5  # Kaggle max is 12h, keep 30 min buffer

def session_time_remaining():
    elapsed = time.time() - SESSION_START
    return (MAX_SESSION_HOURS * 3600) - elapsed

# In the loop:
if session_time_remaining() < 1800:  # Less than 30 min left
    print("Session ending soon — uploading best model to HF Hub...")
    upload_best_to_hub()
    break
```

### Step 17.6: Download AutoWhisper Results

After the Kaggle session ends, pull results locally:

```bash
# Download the best model
huggingface-cli download nishantgaurav23/pasketti-autowhisper-best \
    --local-dir results/autowhisper/best_model

# Download experiment logs
huggingface-cli download nishantgaurav23/pasketti-autowhisper-best \
    --include "logs/*" \
    --local-dir results/autowhisper/logs
```

### Step 17.7: Analyze Results

```bash
# Print summary (total experiments, best WER, improvement over baseline)
python -m src.autowhisper.logger summary results/autowhisper/run_mar17/results.tsv

# Generate progress plot
python -m src.autowhisper.logger plot results/autowhisper/run_mar17/results.tsv
# Output: results/autowhisper/run_mar17/progress.png
```

### AutoWhisper Tips

- **Start with Whisper-small** — faster iteration (15 min/experiment vs. 30 min for large-v3 LoRA)
- **Scripted mode first** — test known hyperparameter variations before spending API credits on agent mode
- **~30 experiments per session** — a 12-hour Kaggle session fits ~30 experiments at 15 min each (with buffer)
- **Check results.tsv** — even discarded experiments teach you what doesn't work
- **Apply winning config manually** — after analysis, copy the best train.py config into your main training scripts for a full training run

---

## 18. Troubleshooting

### CUDA Out of Memory (OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fixes:**
- Reduce `per_device_train_batch_size` to 1 in config
- Increase `gradient_accumulation_steps` proportionally
- Ensure `gradient_checkpointing: true` in config
- For LoRA: ensure `load_in_8bit: true` in config
- Restart the kernel/runtime to clear GPU memory

### bitsandbytes Not Working on macOS

```
RuntimeError: bitsandbytes not supported on this platform
```

**Fix:** Use `--dry-run` mode on MacBook. INT8 quantization requires a CUDA GPU. The dry-run flag skips INT8 automatically.

### Both --noise-dir and --realclass-dir Required

```
ValueError: Both --noise-dir and --realclass-dir must be provided together
```

**Fix:** Either provide both flags or neither. You cannot use just one noise source.

### HuggingFace Hub Authentication Failed

```
huggingface_hub.utils.HfHubHTTPError: 401 Client Error
```

**Fix:**
```bash
huggingface-cli login
# Re-enter your token from https://huggingface.co/settings/tokens
# Ensure the token has "write" permissions
```

### Model Download Slow/Fails

```
ConnectionError: HTTPSConnectionPool
```

**Fix:** Whisper-large-v3 is ~6.2 GB. On slow connections:
```python
# Pre-download in a separate cell
from transformers import WhisperForConditionalGeneration, WhisperProcessor
WhisperProcessor.from_pretrained("openai/whisper-large-v3")
WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
```

### Kaggle Session Died Mid-Training

**Fix:** This is why we push to Hub every 500 steps. Resume by:
1. Start a new Kaggle session
2. The training script will download the base model fresh
3. Unfortunately, you restart training from scratch (HF Trainer doesn't auto-resume from Hub)
4. To resume from a checkpoint, download it from Hub first and pass `--output-dir` pointing to the checkpoint directory

### No Audio Files Found

```
WARNING: 0 valid entries after filtering
```

**Fix:** Check that `--audio-dir` points to the directory containing the FLAC files (not a parent directory). The dataset class looks for files matching the `audio_path` field from the JSONL inside this directory.

---

## 19. Quick Reference: All CLI Commands

### Training Commands

```bash
# Dry run (MacBook, no GPU needed)
python src/train_whisper_small.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio_sample \
  --dry-run --no-push-to-hub

python src/train_whisper_lora.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio_sample \
  --dry-run --no-push-to-hub

# Full training (GPU)
python src/train_whisper_small.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --hub-model-id nishantgaurav23/pasketti-whisper-small

python src/train_whisper_lora.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --hub-model-id nishantgaurav23/pasketti-whisper-lora

# With augmentation (GPU)
python src/train_whisper_small.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --noise-dir data/musan_noise \
  --realclass-dir data/realclass_noise

python src/train_whisper_lora.py \
  --metadata-path data/train_word_transcripts.jsonl \
  --audio-dir data/audio \
  --noise-dir data/musan_noise \
  --realclass-dir data/realclass_noise
```

### Weight Management Commands

```bash
# Login to HuggingFace
huggingface-cli login

# Download weights for submission
huggingface-cli download nishantgaurav23/pasketti-whisper-lora \
  --local-dir submission/model_weights/lora_large_v3

huggingface-cli download nishantgaurav23/pasketti-whisper-small \
  --local-dir submission/model_weights/whisper_small_ft
```

### Submission Commands

```bash
# Validate submission (no zip)
bash scripts/build_submission.sh --dry-run

# Build submission.zip
bash scripts/build_submission.sh

# Custom output path
bash scripts/build_submission.sh --output my_submission.zip
```

### Testing Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocess.py -v
pytest tests/test_dataset.py -v
pytest tests/test_inference.py -v
pytest tests/test_submission.py -v

# Lint
ruff check src/ tests/ submission/
```

---

## Recommended Workflow Summary

```
Step 1  →  Download data from DrivenData
Step 2  →  Upload to Google Drive
Step 3  →  Setup HuggingFace Hub token (Section 16)
Step 4  →  Local dry run on MacBook (both models)
Step 5  →  Kaggle Session 1: Train Whisper-small (~3 hrs)
Step 6  →  Kaggle Session 2: Train Whisper-large-v3 LoRA (~5 hrs)
Step 7  →  (Optional) Kaggle Session 3: Retrain with noise augmentation
Step 8  →  (Optional) Kaggle Session 4+: AutoWhisper experiment loop (Section 17)
Step 9  →  Download weights from HuggingFace Hub
Step 10 →  Local evaluation: compute WER, check per-age breakdown
Step 11 →  Build submission.zip
Step 12 →  Upload to DrivenData
```

# Kaggle CLI Workflow — ChildWhisper

## 1. Setup

### Install
```bash
pip install kaggle
```

### Authenticate
1. Go to https://www.kaggle.com/settings/account
2. Click **Create New Token** — downloads `kaggle.json`
3. Place the file:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Alternative** — environment variables (useful in CI or Colab):
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Verify it works:
```bash
kaggle competitions list
```

---

## 2. Competition Data

### Download competition data
```bash
# Download all competition files
kaggle competitions download -c childrens-word-asr

# Download a specific file
kaggle competitions download -c childrens-word-asr -f train_word_transcripts.jsonl

# Unzip
unzip childrens-word-asr.zip -d data/
```

### Submit results
```bash
kaggle competitions submit \
  -c childrens-word-asr \
  -f submission.zip \
  -m "ensemble v1 — whisper-large-v3 LoRA + whisper-small FT"
```

### Check leaderboard
```bash
kaggle competitions leaderboard -c childrens-word-asr
```

---

## 3. Datasets (Upload Training Data to Kaggle)

Kaggle notebooks can only access Kaggle datasets. Upload your audio and metadata as a private dataset.

### Create a new dataset
```bash
mkdir -p kaggle-datasets/pasketti-audio

# Copy your data files into it
cp data/train_word_transcripts.jsonl kaggle-datasets/pasketti-audio/
cp -r data/audio/ kaggle-datasets/pasketti-audio/

# Initialize dataset metadata
kaggle datasets init -p kaggle-datasets/pasketti-audio/
```

Edit `kaggle-datasets/pasketti-audio/dataset-metadata.json`:
```json
{
  "title": "Pasketti Children ASR Audio",
  "id": "nishantgaurav23/pasketti-audio",
  "licenses": [{ "name": "CC-BY-4.0" }]
}
```

```bash
# Upload (first time)
kaggle datasets create -p kaggle-datasets/pasketti-audio/

# Update with new files
kaggle datasets version -p kaggle-datasets/pasketti-audio/ -m "added TalkBank audio"
```

### Download a dataset
```bash
kaggle datasets download -d nishantgaurav23/pasketti-audio
```

---

## 4. Notebooks / Kernels — Remote GPU Execution

This is the core workflow: **develop locally on MacBook, run on Kaggle T4 GPU**.

### 4.1 Initialize a kernel project

```bash
mkdir -p kaggle-kernels/train-small
cd kaggle-kernels/train-small
kaggle kernels init -p .
```

### 4.2 Configure kernel metadata

Edit `kernel-metadata.json`:
```json
{
  "id": "nishantgaurav23/childwhisper-train-small",
  "title": "ChildWhisper — Train Whisper-small",
  "code_file": "02_train_small.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "nishantgaurav23/pasketti-audio"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

Key fields:
| Field | Value | Notes |
|-------|-------|-------|
| `enable_gpu` | `true` | Allocates T4 GPU |
| `enable_internet` | `true` | Needed for HF Hub checkpoint push |
| `is_private` | `"true"` | Keep training code private |
| `dataset_sources` | your dataset IDs | Available at `/kaggle/input/` |
| `kernel_type` | `"notebook"` or `"script"` | Use notebook for .ipynb, script for .py |

### 4.3 Push and run

```bash
# Upload notebook and start GPU execution
kaggle kernels push -p .
```

### 4.4 Monitor execution

```bash
# Check status (running / complete / error / cancelAcknowledged)
kaggle kernels status nishantgaurav23/childwhisper-train-small
```

Poll in a loop:
```bash
while true; do
  STATUS=$(kaggle kernels status nishantgaurav23/childwhisper-train-small 2>&1)
  echo "$(date '+%H:%M:%S') — $STATUS"
  echo "$STATUS" | grep -q "complete\|error\|cancel" && break
  sleep 60
done
```

### 4.5 Download output

```bash
kaggle kernels output nishantgaurav23/childwhisper-train-small -p ./output/
```

### 4.6 Pull existing kernel code

```bash
kaggle kernels pull nishantgaurav23/childwhisper-train-small -p ./pulled/
```

---

## 5. ChildWhisper-Specific Kernel Configs

### Whisper-small Full Fine-Tune (Phase 2)

`kaggle-kernels/train-small/kernel-metadata.json`:
```json
{
  "id": "nishantgaurav23/childwhisper-train-small",
  "title": "ChildWhisper — Whisper-small Full FT",
  "code_file": "02_train_small.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["nishantgaurav23/pasketti-audio"]
}
```
- GPU time: ~6 hours
- Checkpoint destination: HuggingFace Hub (`nishantgaurav23/pasketti-whisper-small`)

### Whisper-large-v3 LoRA (Phase 3)

`kaggle-kernels/train-lora/kernel-metadata.json`:
```json
{
  "id": "nishantgaurav23/childwhisper-train-lora",
  "title": "ChildWhisper — Whisper-large-v3 LoRA",
  "code_file": "03_train_lora.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["nishantgaurav23/pasketti-audio"]
}
```
- GPU time: ~8 hours
- Checkpoint destination: HuggingFace Hub (`nishantgaurav23/pasketti-whisper-lora`)

### Augmented Re-training (Phase 4)

`kaggle-kernels/train-augmented/kernel-metadata.json`:
```json
{
  "id": "nishantgaurav23/childwhisper-train-augmented",
  "title": "ChildWhisper — Augmented Training",
  "code_file": "04_augmented.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "nishantgaurav23/pasketti-audio",
    "nishantgaurav23/realclass-noise"
  ]
}
```

---

## 6. Development Workflow (End-to-End)

```
MacBook (local)                    Kaggle (remote T4 GPU)
─────────────────                  ──────────────────────
1. Write/edit training code
   in src/ and notebooks/
         │
2. Test with CPU on small
   audio sample (10 files)
         │
3. kaggle kernels push ──────────► 4. Notebook runs on T4 GPU
                                      Training for 6-8 hours
                                      Checkpoints → HF Hub
         │
5. kaggle kernels status ◄─────── (poll until complete)
         │
6. kaggle kernels output ◄─────── 7. Download logs & metrics
         │
8. Analyze results locally
   Update code, repeat
```

### Practical tips

- **Always push checkpoints to HF Hub** during training — Kaggle sessions are ephemeral, all local files are lost after the kernel finishes
- **Set HF token as a Kaggle secret**: Go to Kaggle notebook settings → Add Secret → name: `HF_TOKEN`, value: your token. Access in code:
  ```python
  from kaggle_secrets import UserSecretsClient
  secrets = UserSecretsClient()
  hf_token = secrets.get_secret("HF_TOKEN")
  ```
- **GPU quota**: ~30 hrs/week free. Monitor at https://www.kaggle.com/settings. Plan your training runs: small FT (6h) + LoRA (8h) + augmented (8h) = 22h fits in one week
- **Use `kernel_type: "script"`** if you prefer pushing a `.py` file instead of a notebook
- **Notebook cell timeouts**: Kaggle kills notebooks after 12 hours (GPU) or 9 hours (CPU). Keep training within these limits

---

## 7. CLI Quick Reference

| Task | Command |
|------|---------|
| **Setup** | `pip install kaggle` |
| **Auth** | Place `kaggle.json` in `~/.kaggle/` |
| **Download competition data** | `kaggle competitions download -c <name>` |
| **Submit to competition** | `kaggle competitions submit -c <name> -f <file> -m "msg"` |
| **Create dataset** | `kaggle datasets create -p <dir>` |
| **Update dataset** | `kaggle datasets version -p <dir> -m "msg"` |
| **Download dataset** | `kaggle datasets download -d <owner>/<name>` |
| **Init kernel** | `kaggle kernels init -p .` |
| **Push & run kernel** | `kaggle kernels push -p .` |
| **Check kernel status** | `kaggle kernels status <owner>/<kernel>` |
| **Download kernel output** | `kaggle kernels output <owner>/<kernel> -p <dir>` |
| **Pull kernel code** | `kaggle kernels pull <owner>/<kernel> -p <dir>` |
| **List your kernels** | `kaggle kernels list --mine` |

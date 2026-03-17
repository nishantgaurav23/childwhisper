# Roadmap — ChildWhisper (Pasketti Children's ASR Challenge, Word Track)

## Competition Info
- **Competition**: [DrivenData — On Top of Pasketti Word Track](https://www.drivendata.org/competitions/308/childrens-word-asr/)
- **Deadline**: ~Early April 2026
- **Prize Pool**: $70,000 (Word Track) + $20,000 (Noisy WER bonuses)
- **Submissions**: 3 per rolling 7-day window
- **Target WER**: <0.20 (beating 0.2370 benchmark)

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11 | Matches competition runtime |
| ASR Model (Primary) | Whisper-large-v3 + LoRA | Best WER with parameter-efficient fine-tuning; fits T4 via INT8 |
| ASR Model (Secondary) | Whisper-small (full FT) | Fast iteration + less hallucination on short clips; ensemble diversity |
| PEFT | HuggingFace PEFT (LoRA) | Deepest ecosystem, most community examples |
| Quantization | bitsandbytes INT8 | Fits large-v3 training on T4 (16GB) |
| Audio | librosa + soundfile | Standard, reliable, 16kHz resampling |
| Augmentation | audiomentations | Classroom noise mixing, SNR control |
| Evaluation | jiwer + Whisper EnglishTextNormalizer | Competition-standard WER computation |
| Training GPU | Kaggle T4 (30 hrs/wk free) | $0, sufficient for LoRA + small FT |
| Inference GPU | A100 80GB (competition) | Fits both models in fp16 |
| Model Storage | HuggingFace Hub (private) | Free, persistent across Kaggle sessions |
| Testing | pytest | Simple, sufficient |
| Linting | ruff (line-length: 100) | Fast, modern |

## Budget Estimate

| Item | Monthly Cost |
|------|-------------|
| Kaggle GPU (30 hrs/wk) | $0 |
| Colab GPU (~20 hrs/wk) | $0 |
| HuggingFace Hub (private) | $0 |
| Colab Pro (optional) | $0-10 |
| **Total** | **$0-10** |

## Phases Overview

| Phase | Name | Specs | Key Output |
|-------|------|-------|------------|
| 1 | Project Setup & Baseline | 5 | Working zero-shot submission, local validation |
| 2 | Whisper-small Fine-Tune | 4 | Fine-tuned small model, WER ~0.15-0.20 |
| 3 | Whisper-large-v3 LoRA | 4 | LoRA adapter, ensemble inference, WER ~0.12-0.17 |
| 4 | Noise Augmentation | 3 | Noise-robust models, improved Noisy WER |
| 5 | Optimizations & Polish | 4 | Final WER squeeze, submission packaging |

---

## Phase 1: Project Setup & Zero-Shot Baseline (1-2 days)

**Goal**: Working submission with zero training. Establish baseline WER.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S1.1 | specs/spec-S1.1-project-structure/ | — | project root | Project structure & dependencies | requirements.txt, .gitignore, configs | done |
| S1.2 | specs/spec-S1.2-audio-preprocessing/ | S1.1 | src/preprocess.py | Audio preprocessing pipeline | 16kHz resample, trim, duration filter, silence detection | done |
| S1.3 | specs/spec-S1.3-text-normalization/ | S1.1 | src/utils.py | Text normalization | Whisper EnglishTextNormalizer wrapper | done |
| S1.4 | specs/spec-S1.4-inference-pipeline/ | S1.2, S1.3 | submission/main.py | Zero-shot inference pipeline | Whisper-small zero-shot, batch inference, JSONL output | done |
| S1.5 | specs/spec-S1.5-local-validation/ | S1.2, S1.3 | src/evaluate.py | Local validation framework | 90/10 child_id split, per-age WER, noisy val set | done |

---

## Phase 2: Whisper-small Full Fine-Tune (3-4 days)

**Goal**: Fine-tune smallest viable model end-to-end. First real WER improvement.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S2.1 | specs/spec-S2.1-dataset-class/ | S1.2, S1.3 | src/dataset.py | PyTorch Dataset for Whisper | Tokenization, feature extraction, padding, collation | done |
| S2.2 | specs/spec-S2.2-train-small/ | S2.1, S1.5 | src/train_whisper_small.py | Whisper-small training script | Full FT, SpecAugment, gradient ckpt, HF Hub push | done |
| S2.3 | specs/spec-S2.3-kaggle-notebook-small/ | S2.2 | notebooks/02_train_small.ipynb | Kaggle training notebook | Kaggle-ready, data upload, checkpoint management | done |
| S2.4 | specs/spec-S2.4-inference-finetuned/ | S2.2, S1.4 | submission/main.py | Inference with fine-tuned small | Update main.py to load fine-tuned weights | done |

---

## Phase 3: Whisper-large-v3 LoRA (4-5 days)

**Goal**: Add larger model with parameter-efficient fine-tuning.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S3.1 | specs/spec-S3.1-lora-config/ | S2.1 | src/train_whisper_lora.py | LoRA configuration & training | r=32, alpha=64, INT8, q_proj+v_proj | done |
| S3.2 | specs/spec-S3.2-kaggle-notebook-lora/ | S3.1 | notebooks/03_train_lora.ipynb | Kaggle LoRA training notebook | INT8 loading, LoRA training, adapter save | done |
| S3.3 | specs/spec-S3.3-ensemble-inference/ | S3.1, S2.4 | submission/main.py | Ensemble inference pipeline | Both models sequential, confidence merge, time budget | done |
| S3.4 | specs/spec-S3.4-submission-packaging/ | S3.3 | scripts/build_submission.sh | Submission zip builder | Bundle weights + code, size check, Docker test | done |

---

## Phase 4: Noise Augmentation (3-4 days)

**Goal**: Improve Noisy WER for classroom bonus prize.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S4.1 | specs/spec-S4.1-augmentation-pipeline/ | S2.1 | src/augment.py | Noise augmentation pipeline | RealClass + MUSAN mixing, SNR 0-20dB, 50/20/30 split | done |
| S4.2 | specs/spec-S4.2-noisy-validation/ | S4.1, S1.5 | src/evaluate.py | Noisy validation set | Synthetic noisy val (val audio + RealClass SNR 10dB) | pending |
| S4.3 | specs/spec-S4.3-retrain-augmented/ | S4.1, S3.1 | notebooks/04_augmented.ipynb | Retrain with augmented data | Re-train LoRA + small on augmented data | pending |

---

## Phase 5: Optimizations & Polish (2-3 days)

**Goal**: Squeeze final WER improvements before deadline.

| Spec | Spec Location | Depends On | Location | Feature | Notes | Status |
|------|--------------|------------|----------|---------|-------|--------|
| S5.1 | specs/spec-S5.1-post-processing/ | S3.3 | src/utils.py | Post-processing corrections | Children's vocab spell-check, common ASR error fixes | pending |
| S5.2 | specs/spec-S5.2-faster-inference/ | S3.3 | submission/main.py | Inference optimization | CTranslate2/faster-whisper, larger beam or more passes | pending |
| S5.3 | specs/spec-S5.3-error-analysis/ | S1.5, S3.3 | src/evaluate.py, notebooks/01_eda.ipynb | Error analysis tooling | Per-age WER, sub/ins/del breakdown, hallucination detection | pending |
| S5.4 | specs/spec-S5.4-final-submission/ | S5.1, S5.2 | submission/ | Final submission package | Docker test, size verify, competition submit | pending |

---

## Master Spec Index

| Spec | Feature | Phase | Depends On | Status |
|------|---------|-------|------------|--------|
| S1.1 | Project structure & dependencies | 1 | — | done |
| S1.2 | Audio preprocessing pipeline | 1 | S1.1 | done |
| S1.3 | Text normalization | 1 | S1.1 | done |
| S1.4 | Zero-shot inference pipeline | 1 | S1.2, S1.3 | done |
| S1.5 | Local validation framework | 1 | S1.2, S1.3 | done |
| S2.1 | PyTorch Dataset for Whisper | 2 | S1.2, S1.3 | done |
| S2.2 | Whisper-small training script | 2 | S2.1, S1.5 | done |
| S2.3 | Kaggle training notebook (small) | 2 | S2.2 | done |
| S2.4 | Inference with fine-tuned small | 2 | S2.2, S1.4 | done |
| S3.1 | LoRA configuration & training | 3 | S2.1 | done |
| S3.2 | Kaggle LoRA training notebook | 3 | S3.1 | done |
| S3.3 | Ensemble inference pipeline | 3 | S3.1, S2.4 | done |
| S3.4 | Submission zip builder | 3 | S3.3 | done |
| S4.1 | Noise augmentation pipeline | 4 | S2.1 | done |
| S4.2 | Noisy validation set | 4 | S4.1, S1.5 | pending |
| S4.3 | Retrain with augmented data | 4 | S4.1, S3.1 | pending |
| S5.1 | Post-processing corrections | 5 | S3.3 | pending |
| S5.2 | Inference optimization | 5 | S3.3 | pending |
| S5.3 | Error analysis tooling | 5 | S1.5, S3.3 | pending |
| S5.4 | Final submission package | 5 | S5.1, S5.2 | pending |

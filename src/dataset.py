"""PyTorch Dataset for Whisper fine-tuning on children's speech.

Provides WhisperDataset (audio loading + feature extraction + tokenization),
WhisperDataCollator (batched padding), and train/val split by child_id.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from src.preprocess import (
    is_silence,
    is_valid_duration,
    load_audio,
    load_metadata,
    trim_silence,
)
from src.utils import normalize_text


class WhisperDataset(Dataset):
    """PyTorch Dataset that loads audio, extracts Whisper features, and tokenizes transcripts."""

    def __init__(
        self,
        metadata_path: str | Path,
        audio_dir: str | Path,
        model_name: str = "openai/whisper-small",
        augment_fn: Callable | None = None,
        min_duration: float = 0.3,
        max_duration: float = 30.0,
        language: str = "en",
        task: str = "transcribe",
    ):
        self.audio_dir = Path(audio_dir)
        self.augment_fn = augment_fn
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.processor.tokenizer.set_prefix_tokens(language=language, task=task)

        raw_metadata = load_metadata(metadata_path)
        self.entries = self._filter_entries(raw_metadata)

    def _filter_entries(self, metadata: list[dict]) -> list[dict]:
        """Filter out entries with missing audio or invalid duration."""
        valid = []
        for entry in metadata:
            audio_path = self.audio_dir / entry["audio_path"]
            if not audio_path.exists():
                continue
            dur = entry.get("audio_duration_sec")
            if dur is not None and not is_valid_duration(
                dur, min_dur=self.min_duration, max_dur=self.max_duration
            ):
                continue
            valid.append(entry)
        return valid

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        audio_path = self.audio_dir / entry["audio_path"]

        audio, sr = load_audio(audio_path)
        audio = trim_silence(audio)

        if is_silence(audio):
            transcript = ""
        else:
            transcript = normalize_text(entry.get("orthographic_text", ""))

        if self.augment_fn is not None:
            audio = self.augment_fn(audio, sample_rate=sr)

        features = self.processor.feature_extractor(
            audio, sampling_rate=sr, return_tensors="np"
        )
        input_features = features.input_features[0]

        labels = self.processor.tokenizer(transcript)["input_ids"]

        return {"input_features": input_features, "labels": labels}


class WhisperDataCollator:
    """Data collator that pads input_features and labels for Whisper training."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [f["input_features"] for f in features]
        label_lists = [f["labels"] for f in features]

        # Stack input features (all same shape from feature extractor)
        input_features = torch.tensor(np.array(input_features), dtype=torch.float32)

        # Pad labels to max length, using -100 for padding
        max_len = max(len(lab) for lab in label_lists)
        padded_labels = []
        for lab in label_lists:
            padded = list(lab) + [-100] * (max_len - len(lab))
            padded_labels.append(padded)
        labels = torch.tensor(padded_labels, dtype=torch.long)

        return {"input_features": input_features, "labels": labels}


def create_train_val_split(
    metadata: list[dict],
    val_ratio: float = 0.1,
    split_by: str = "child_id",
    stratify_by: str = "age_bucket",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split metadata by child_id with age_bucket stratification.

    No child appears in both train and val. Stratification ensures
    proportional age bucket representation.
    """
    rng = random.Random(seed)

    # Group children by age bucket
    bucket_to_children: dict[str, set[str]] = defaultdict(set)
    child_to_entries: dict[str, list[dict]] = defaultdict(list)

    for entry in metadata:
        child = entry[split_by]
        bucket = entry.get(stratify_by, "unknown")
        bucket_to_children[bucket].add(child)
        child_to_entries[child].append(entry)

    val_children: set[str] = set()

    # Stratified selection: pick val_ratio of children from each bucket
    for bucket, children in bucket_to_children.items():
        children_list = sorted(children)
        rng.shuffle(children_list)
        n_val = max(1, round(len(children_list) * val_ratio))
        val_children.update(children_list[:n_val])

    train_entries = []
    val_entries = []
    for entry in metadata:
        if entry[split_by] in val_children:
            val_entries.append(entry)
        else:
            train_entries.append(entry)

    return train_entries, val_entries

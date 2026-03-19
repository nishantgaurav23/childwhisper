"""Tests for src/dataset.py — WhisperDataset, WhisperDataCollator, train/val split."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.dataset import (
    WhisperDataCollator, WhisperDataset, create_train_val_split, stratified_subset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_metadata(audio_dir: Path, entries: list[dict]) -> Path:
    """Write JSONL metadata and return path."""
    jsonl_path = audio_dir.parent / "metadata.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return jsonl_path


@pytest.fixture
def sample_data(tmp_path):
    """Create a temporary dataset with 5 audio files and metadata."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    entries = []
    for i in range(5):
        fname = f"utt_{i}.wav"
        (audio_dir / fname).touch()  # dummy file for exists() check
        entries.append({
            "utterance_id": f"utt_{i}",
            "child_id": f"child_{i % 3}",
            "session_id": f"session_{i}",
            "audio_path": f"audio/{fname}",
            "audio_duration_sec": 1.0 + i * 0.5,
            "age_bucket": ["3-4", "5-7", "8-11", "5-7", "3-4"][i],
            "orthographic_text": f"hello world {i}",
        })

    jsonl_path = _make_metadata(audio_dir, entries)
    return tmp_path, jsonl_path, entries


@pytest.fixture
def mock_processor():
    """Create a mock WhisperProcessor."""
    processor = MagicMock()
    # Feature extractor returns tensor-like
    feature_result = MagicMock()
    feature_result.input_features = [np.random.randn(80, 3000).astype(np.float32)]
    processor.feature_extractor.return_value = feature_result
    # Tokenizer returns token IDs
    processor.tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}
    processor.tokenizer.pad_token_id = 0
    # set_prefix_tokens
    processor.tokenizer.set_prefix_tokens = MagicMock()
    return processor


# ---------------------------------------------------------------------------
# Tests: WhisperDataset
# ---------------------------------------------------------------------------

class TestWhisperDataset:
    @patch("src.dataset.WhisperProcessor.from_pretrained")
    def test_dataset_init(self, mock_from_pretrained, sample_data, mock_processor):
        """Dataset loads metadata, filters invalid entries, stores correct count."""
        mock_from_pretrained.return_value = mock_processor
        data_dir, jsonl_path, entries = sample_data

        ds = WhisperDataset(
            metadata_path=jsonl_path,
            audio_dir=data_dir,
            model_name="openai/whisper-small",
        )
        assert len(ds) == 5
        mock_from_pretrained.assert_called_once_with("openai/whisper-small")

    @patch("src.dataset.is_silence", return_value=False)
    @patch("src.dataset.trim_silence", side_effect=lambda a, **kw: a)
    @patch("src.dataset.load_audio")
    @patch("src.dataset.WhisperProcessor.from_pretrained")
    def test_dataset_getitem(
        self, mock_from_pretrained, mock_load, mock_trim, mock_silence,
        sample_data, mock_processor,
    ):
        """Returns dict with input_features and labels."""
        mock_from_pretrained.return_value = mock_processor
        mock_load.return_value = (np.random.randn(16000).astype(np.float32), 16000)
        data_dir, jsonl_path, _ = sample_data

        ds = WhisperDataset(
            metadata_path=jsonl_path,
            audio_dir=data_dir,
            model_name="openai/whisper-small",
        )
        item = ds[0]
        assert "input_features" in item
        assert "labels" in item
        assert isinstance(item["input_features"], np.ndarray)
        assert isinstance(item["labels"], list)

    @patch("src.dataset.is_silence", return_value=True)
    @patch("src.dataset.trim_silence", side_effect=lambda a, **kw: a)
    @patch("src.dataset.load_audio")
    @patch("src.dataset.WhisperProcessor.from_pretrained")
    def test_dataset_silence_handling(
        self, mock_from_pretrained, mock_load, mock_trim, mock_silence,
        mock_processor, tmp_path,
    ):
        """Silent audio returns empty transcript tokens."""
        mock_from_pretrained.return_value = mock_processor
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        # Create a dummy file so it passes the exists() check at init
        (audio_dir / "silent.wav").touch()

        entries = [{
            "utterance_id": "silent_utt",
            "child_id": "child_0",
            "session_id": "s0",
            "audio_path": "audio/silent.wav",
            "audio_duration_sec": 1.0,
            "age_bucket": "5-7",
            "orthographic_text": "some words",
        }]
        jsonl_path = _make_metadata(audio_dir, entries)

        # Mock tokenizer to return empty for empty text
        mock_processor.tokenizer.side_effect = lambda text, **kw: (
            {"input_ids": []} if text == "" else {"input_ids": [1, 2, 3]}
        )

        ds = WhisperDataset(
            metadata_path=jsonl_path,
            audio_dir=tmp_path,
            model_name="openai/whisper-small",
        )
        item = ds[0]
        assert item["labels"] == []

    @patch("src.dataset.WhisperProcessor.from_pretrained")
    def test_dataset_missing_audio(self, mock_from_pretrained, mock_processor, tmp_path):
        """Missing audio files are filtered out at init."""
        mock_from_pretrained.return_value = mock_processor
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "exists.wav").touch()

        entries = [
            {
                "utterance_id": "utt_ok",
                "child_id": "c1",
                "session_id": "s1",
                "audio_path": "audio/exists.wav",
                "audio_duration_sec": 1.0,
                "age_bucket": "5-7",
                "orthographic_text": "hello",
            },
            {
                "utterance_id": "utt_missing",
                "child_id": "c2",
                "session_id": "s2",
                "audio_path": "audio/missing.wav",
                "audio_duration_sec": 1.0,
                "age_bucket": "5-7",
                "orthographic_text": "bye",
            },
        ]
        jsonl_path = _make_metadata(audio_dir, entries)

        ds = WhisperDataset(
            metadata_path=jsonl_path,
            audio_dir=tmp_path,
            model_name="openai/whisper-small",
        )
        assert len(ds) == 1

    @patch("src.dataset.is_silence", return_value=False)
    @patch("src.dataset.trim_silence", side_effect=lambda a, **kw: a)
    @patch("src.dataset.load_audio")
    @patch("src.dataset.WhisperProcessor.from_pretrained")
    def test_dataset_augmentation(
        self, mock_from_pretrained, mock_load, mock_trim, mock_silence,
        sample_data, mock_processor,
    ):
        """When augment callable provided, it's applied to audio."""
        mock_from_pretrained.return_value = mock_processor
        mock_load.return_value = (np.random.randn(16000).astype(np.float32), 16000)
        data_dir, jsonl_path, _ = sample_data

        augment_fn = MagicMock(side_effect=lambda audio, sample_rate: audio * 0.5)

        ds = WhisperDataset(
            metadata_path=jsonl_path,
            audio_dir=data_dir,
            model_name="openai/whisper-small",
            augment_fn=augment_fn,
        )
        _ = ds[0]
        augment_fn.assert_called_once()

    @patch("src.dataset.WhisperProcessor.from_pretrained")
    def test_dataset_duration_filter(self, mock_from_pretrained, mock_processor, tmp_path):
        """Entries with out-of-range duration are filtered at init."""
        mock_from_pretrained.return_value = mock_processor
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "short.wav").touch()
        (audio_dir / "ok.wav").touch()

        entries = [
            {
                "utterance_id": "too_short",
                "child_id": "c1",
                "session_id": "s1",
                "audio_path": "audio/short.wav",
                "audio_duration_sec": 0.1,
                "age_bucket": "5-7",
                "orthographic_text": "hi",
            },
            {
                "utterance_id": "ok_dur",
                "child_id": "c2",
                "session_id": "s2",
                "audio_path": "audio/ok.wav",
                "audio_duration_sec": 2.0,
                "age_bucket": "5-7",
                "orthographic_text": "hello world",
            },
        ]
        jsonl_path = _make_metadata(audio_dir, entries)

        ds = WhisperDataset(
            metadata_path=jsonl_path,
            audio_dir=tmp_path,
            model_name="openai/whisper-small",
            min_duration=0.3,
            max_duration=30.0,
        )
        assert len(ds) == 1

    @patch("src.dataset.WhisperProcessor.from_pretrained")
    def test_dataset_with_processor(self, mock_from_pretrained, sample_data, mock_processor):
        """Verifies correct WhisperProcessor model name loading."""
        mock_from_pretrained.return_value = mock_processor
        data_dir, jsonl_path, _ = sample_data

        WhisperDataset(
            metadata_path=jsonl_path,
            audio_dir=data_dir,
            model_name="openai/whisper-large-v3",
        )
        mock_from_pretrained.assert_called_once_with("openai/whisper-large-v3")


# ---------------------------------------------------------------------------
# Tests: WhisperDataCollator
# ---------------------------------------------------------------------------

class TestWhisperDataCollator:
    def test_collator_padding(self):
        """Collator pads features and labels correctly, labels padded with -100."""
        features = [
            {
                "input_features": np.random.randn(80, 3000).astype(np.float32),
                "labels": [1, 2, 3],
            },
            {
                "input_features": np.random.randn(80, 3000).astype(np.float32),
                "labels": [4, 5],
            },
        ]

        collator = WhisperDataCollator(pad_token_id=0)
        batch = collator(features)

        assert "input_features" in batch
        assert "labels" in batch
        assert batch["input_features"].shape[0] == 2
        assert batch["labels"].shape[0] == 2
        # Check padding value for labels
        assert batch["labels"].shape[1] == 3  # max label length
        assert batch["labels"][1, 2].item() == -100  # padded position

    def test_collator_batch_shapes(self):
        """Output batch has consistent tensor shapes."""
        features = [
            {
                "input_features": np.random.randn(80, 3000).astype(np.float32),
                "labels": [1, 2, 3, 4, 5],
            },
            {
                "input_features": np.random.randn(80, 3000).astype(np.float32),
                "labels": [6, 7, 8],
            },
            {
                "input_features": np.random.randn(80, 3000).astype(np.float32),
                "labels": [9],
            },
        ]

        collator = WhisperDataCollator(pad_token_id=0)
        batch = collator(features)

        assert batch["input_features"].shape == (3, 80, 3000)
        assert batch["labels"].shape == (3, 5)
        assert isinstance(batch["input_features"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)


# ---------------------------------------------------------------------------
# Tests: create_train_val_split
# ---------------------------------------------------------------------------

class TestTrainValSplit:
    def _make_split_metadata(self, n_children=20, utterances_per_child=5):
        """Generate metadata for split testing."""
        entries = []
        age_buckets = ["3-4", "5-7", "8-11", "12+"]
        for c in range(n_children):
            child_id = f"child_{c}"
            age = age_buckets[c % len(age_buckets)]
            for u in range(utterances_per_child):
                entries.append({
                    "utterance_id": f"utt_{c}_{u}",
                    "child_id": child_id,
                    "session_id": f"session_{c}",
                    "audio_path": f"audio/utt_{c}_{u}.wav",
                    "audio_duration_sec": 2.0,
                    "age_bucket": age,
                    "orthographic_text": f"word {c} {u}",
                })
        return entries

    def test_train_val_split_no_leakage(self):
        """No child_id appears in both train and val sets."""
        metadata = self._make_split_metadata(n_children=20)
        train, val = create_train_val_split(metadata, val_ratio=0.1, seed=42)

        train_children = {e["child_id"] for e in train}
        val_children = {e["child_id"] for e in val}
        assert len(train_children & val_children) == 0

    def test_train_val_split_stratification(self):
        """Age buckets represented proportionally in both splits."""
        metadata = self._make_split_metadata(n_children=40)
        train, val = create_train_val_split(metadata, val_ratio=0.2, seed=42)

        def bucket_dist(data):
            buckets = {}
            for e in data:
                b = e["age_bucket"]
                buckets[b] = buckets.get(b, 0) + 1
            total = len(data)
            return {b: c / total for b, c in buckets.items()}

        train_dist = bucket_dist(train)
        val_dist = bucket_dist(val)

        # Each bucket should be within 15% of original distribution
        for bucket in train_dist:
            if bucket in val_dist:
                assert abs(train_dist[bucket] - val_dist[bucket]) < 0.15, (
                    f"Bucket {bucket}: train={train_dist[bucket]:.2f}, val={val_dist[bucket]:.2f}"
                )

    def test_train_val_split_deterministic(self):
        """Same seed produces same split."""
        metadata = self._make_split_metadata(n_children=20)
        train1, val1 = create_train_val_split(metadata, val_ratio=0.1, seed=42)
        train2, val2 = create_train_val_split(metadata, val_ratio=0.1, seed=42)

        assert [e["utterance_id"] for e in train1] == [e["utterance_id"] for e in train2]
        assert [e["utterance_id"] for e in val1] == [e["utterance_id"] for e in val2]

    def test_train_val_split_coverage(self):
        """All utterances appear in exactly one split."""
        metadata = self._make_split_metadata(n_children=20)
        train, val = create_train_val_split(metadata, val_ratio=0.1, seed=42)

        all_ids = {e["utterance_id"] for e in metadata}
        split_ids = {e["utterance_id"] for e in train} | {e["utterance_id"] for e in val}
        assert all_ids == split_ids

    def test_train_val_split_ratio(self):
        """Val set is approximately the right size."""
        metadata = self._make_split_metadata(n_children=20)
        train, val = create_train_val_split(metadata, val_ratio=0.1, seed=42)

        total = len(train) + len(val)
        val_ratio = len(val) / total
        # Allow tolerance since we split by child_id, not by utterance
        assert 0.05 <= val_ratio <= 0.25


# ---------------------------------------------------------------------------
# Tests: stratified_subset
# ---------------------------------------------------------------------------

class TestStratifiedSubset:
    def _make_metadata(self, n_children=20, utterances_per_child=5):
        entries = []
        age_buckets = ["3-4", "5-7", "8-11", "12+"]
        for c in range(n_children):
            child_id = f"child_{c}"
            age = age_buckets[c % len(age_buckets)]
            for u in range(utterances_per_child):
                entries.append({
                    "utterance_id": f"utt_{c}_{u}",
                    "child_id": child_id,
                    "audio_path": f"audio/utt_{c}_{u}.wav",
                    "audio_duration_sec": 2.0,
                    "age_bucket": age,
                    "orthographic_text": f"word {c} {u}",
                })
        return entries

    def test_full_fraction_returns_all(self):
        """fraction=1.0 returns original data unchanged."""
        metadata = self._make_metadata(n_children=20)
        result = stratified_subset(metadata, fraction=1.0)
        assert len(result) == len(metadata)

    def test_subset_reduces_data(self):
        """fraction=0.3 returns roughly 30% of data."""
        metadata = self._make_metadata(n_children=40)
        result = stratified_subset(metadata, fraction=0.3)
        assert len(result) < len(metadata)
        # Should be roughly 30% (within tolerance due to child grouping)
        ratio = len(result) / len(metadata)
        assert 0.15 <= ratio <= 0.50

    def test_no_child_leakage(self):
        """All utterances from a child are either fully included or excluded."""
        metadata = self._make_metadata(n_children=40, utterances_per_child=10)
        result = stratified_subset(metadata, fraction=0.3)

        child_counts = {}
        for e in result:
            child_counts[e["child_id"]] = child_counts.get(e["child_id"], 0) + 1

        # Each selected child should have exactly 10 utterances
        for child_id, count in child_counts.items():
            assert count == 10, f"{child_id} has {count} utterances, expected 10"

    def test_preserves_age_distribution(self):
        """Each age bucket is represented in the subset."""
        metadata = self._make_metadata(n_children=40)
        result = stratified_subset(metadata, fraction=0.3)

        original_buckets = {e["age_bucket"] for e in metadata}
        subset_buckets = {e["age_bucket"] for e in result}
        assert original_buckets == subset_buckets

    def test_deterministic(self):
        """Same seed produces same subset."""
        metadata = self._make_metadata(n_children=40)
        r1 = stratified_subset(metadata, fraction=0.3, seed=42)
        r2 = stratified_subset(metadata, fraction=0.3, seed=42)
        assert [e["utterance_id"] for e in r1] == [e["utterance_id"] for e in r2]

    def test_different_seeds_differ(self):
        """Different seeds produce different subsets."""
        metadata = self._make_metadata(n_children=40)
        r1 = stratified_subset(metadata, fraction=0.3, seed=42)
        r2 = stratified_subset(metadata, fraction=0.3, seed=99)
        ids1 = {e["utterance_id"] for e in r1}
        ids2 = {e["utterance_id"] for e in r2}
        assert ids1 != ids2

    def test_rare_bucket_keeps_at_least_one(self):
        """A bucket with only 1 child still gets included."""
        metadata = self._make_metadata(n_children=20)
        # Add a rare bucket with just 1 child
        metadata.append({
            "utterance_id": "rare_utt",
            "child_id": "rare_child",
            "audio_path": "audio/rare.wav",
            "audio_duration_sec": 2.0,
            "age_bucket": "rare_bucket",
            "orthographic_text": "rare",
        })
        result = stratified_subset(metadata, fraction=0.3)
        rare = [e for e in result if e["age_bucket"] == "rare_bucket"]
        assert len(rare) == 1

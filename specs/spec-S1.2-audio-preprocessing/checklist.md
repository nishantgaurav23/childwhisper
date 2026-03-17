# Checklist S1.2 — Audio Preprocessing Pipeline

## Phase 1: Red (Write Tests)
- [x] Write test_load_audio — resamples to 16kHz mono
- [x] Write test_trim_silence — trims leading/trailing silence
- [x] Write test_is_silence — detects silent audio below threshold
- [x] Write test_is_silence_non_silent — returns False for audio above threshold
- [x] Write test_get_duration — returns correct duration in seconds
- [x] Write test_is_valid_duration — accepts valid, rejects too short/long
- [x] Write test_preprocess_utterance_valid — full pipeline happy path
- [x] Write test_preprocess_utterance_too_short — returns None
- [x] Write test_preprocess_utterance_too_long — returns None
- [x] Write test_preprocess_utterance_empty_transcript — returns None
- [x] Write test_preprocess_utterance_silence — returns empty transcript
- [x] Write test_load_metadata — parses JSONL correctly
- [x] Verify all tests FAIL (no implementation yet)

## Phase 2: Green (Implement)
- [x] Implement load_audio
- [x] Implement trim_silence
- [x] Implement is_silence
- [x] Implement get_duration
- [x] Implement is_valid_duration
- [x] Implement preprocess_utterance
- [x] Implement load_metadata
- [x] Verify all tests PASS

## Phase 3: Refactor
- [x] Run ruff, fix any issues
- [x] Verify all tests still pass
- [x] Review for code clarity and adherence to standards

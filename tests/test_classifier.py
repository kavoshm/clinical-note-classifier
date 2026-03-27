"""
Tests for the classifier engine (src/classifier.py).
Covers retry logic, batch processing error handling, and safety-first defaults.
Uses mocking to avoid real OpenAI API calls.
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from openai import APIError, RateLimitError, APITimeoutError
from pydantic import ValidationError

from src.classifier import ClinicalNoteClassifier, load_notes_from_file
from src.models import ClassificationResult, ClassifiedNote, ClinicalNote


# =============================================================================
# Fixtures
# =============================================================================

VALID_API_RESPONSE = json.dumps({
    "urgency_level": 3,
    "primary_complaint": "Acute appendicitis, uncomplicated",
    "icd10_code": "K35.80",
    "icd10_description": "Unspecified acute appendicitis without abscess",
    "reasoning": "CT-confirmed acute appendicitis without perforation. Patient hemodynamically stable.",
    "recommended_action": "Surgical consult for appendectomy, NPO status, IV fluids, pain management",
})

INVALID_JSON_RESPONSE = "This is not valid JSON at all."

INVALID_SCHEMA_RESPONSE = json.dumps({
    "urgency_level": 99,  # Out of range
    "primary_complaint": "Test",
    "icd10_code": "K35.80",
    "icd10_description": "Test",
    "reasoning": "Test reasoning",
    "recommended_action": "Test action",
})


def _make_mock_response(content: str) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


@pytest.fixture
def mock_openai_client():
    """Patch OpenAI client so no real API calls are made."""
    with patch("src.classifier.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


# =============================================================================
# Single Note Classification Tests
# =============================================================================

class TestClassifyNote:
    """Tests for ClinicalNoteClassifier.classify_note()."""

    def test_successful_classification(self, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = _make_mock_response(
            VALID_API_RESPONSE
        )
        classifier = ClinicalNoteClassifier()
        result = classifier.classify_note("Test clinical note with enough content.")
        assert isinstance(result, ClassificationResult)
        assert result.urgency_level == 3
        assert result.icd10_code == "K35.80"

    def test_api_called_with_correct_params(self, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value = _make_mock_response(
            VALID_API_RESPONSE
        )
        classifier = ClinicalNoteClassifier(model="gpt-4o-mini", temperature=0.0)
        classifier.classify_note("Some note.")

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"


# =============================================================================
# Retry Logic Tests
# =============================================================================

class TestRetryLogic:
    """Tests for retry behavior on transient failures."""

    @patch("src.classifier.time.sleep")
    def test_retry_on_json_decode_error(self, mock_sleep, mock_openai_client):
        """Should retry on malformed JSON, then succeed."""
        mock_openai_client.chat.completions.create.side_effect = [
            _make_mock_response(INVALID_JSON_RESPONSE),
            _make_mock_response(VALID_API_RESPONSE),
        ]
        classifier = ClinicalNoteClassifier(max_retries=3, retry_delay=0.01)
        result = classifier.classify_note("Test note.")
        assert result.urgency_level == 3
        assert mock_openai_client.chat.completions.create.call_count == 2

    @patch("src.classifier.time.sleep")
    def test_retry_on_validation_error(self, mock_sleep, mock_openai_client):
        """Should retry on Pydantic validation failure, then succeed."""
        mock_openai_client.chat.completions.create.side_effect = [
            _make_mock_response(INVALID_SCHEMA_RESPONSE),
            _make_mock_response(VALID_API_RESPONSE),
        ]
        classifier = ClinicalNoteClassifier(max_retries=3, retry_delay=0.01)
        result = classifier.classify_note("Test note.")
        assert result.urgency_level == 3

    @patch("src.classifier.time.sleep")
    def test_raises_after_max_retries_json_error(self, mock_sleep, mock_openai_client):
        """Should raise RuntimeError after exhausting retries on JSON errors."""
        mock_openai_client.chat.completions.create.return_value = _make_mock_response(
            INVALID_JSON_RESPONSE
        )
        classifier = ClinicalNoteClassifier(max_retries=3, retry_delay=0.01)
        with pytest.raises(RuntimeError, match="Classification failed after 3 attempts"):
            classifier.classify_note("Test note.")
        assert mock_openai_client.chat.completions.create.call_count == 3

    @patch("src.classifier.time.sleep")
    def test_retry_on_rate_limit_error(self, mock_sleep, mock_openai_client):
        """Should retry with exponential backoff on rate limits."""
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {}

        mock_openai_client.chat.completions.create.side_effect = [
            RateLimitError(
                message="Rate limit exceeded",
                response=rate_limit_response,
                body=None,
            ),
            _make_mock_response(VALID_API_RESPONSE),
        ]
        classifier = ClinicalNoteClassifier(max_retries=3, retry_delay=0.01)
        result = classifier.classify_note("Test note.")
        assert result.urgency_level == 3

    @patch("src.classifier.time.sleep")
    def test_raises_after_max_retries_rate_limit(self, mock_sleep, mock_openai_client):
        """Should raise RuntimeError after exhausting retries on rate limits."""
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {}

        mock_openai_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=rate_limit_response,
            body=None,
        )
        classifier = ClinicalNoteClassifier(max_retries=2, retry_delay=0.01)
        with pytest.raises(RuntimeError, match="Rate limited"):
            classifier.classify_note("Test note.")

    @patch("src.classifier.time.sleep")
    def test_retry_on_api_timeout(self, mock_sleep, mock_openai_client):
        """Should retry on API timeout errors."""
        mock_openai_client.chat.completions.create.side_effect = [
            APITimeoutError(request=MagicMock()),
            _make_mock_response(VALID_API_RESPONSE),
        ]
        classifier = ClinicalNoteClassifier(max_retries=3, retry_delay=0.01)
        result = classifier.classify_note("Test note.")
        assert result.urgency_level == 3

    @patch("src.classifier.time.sleep")
    def test_retry_on_api_error(self, mock_sleep, mock_openai_client):
        """Should retry on generic API errors."""
        mock_openai_client.chat.completions.create.side_effect = [
            APIError(
                message="Server error",
                request=MagicMock(),
                body=None,
            ),
            _make_mock_response(VALID_API_RESPONSE),
        ]
        classifier = ClinicalNoteClassifier(max_retries=3, retry_delay=0.01)
        result = classifier.classify_note("Test note.")
        assert result.urgency_level == 3

    @patch("src.classifier.time.sleep")
    def test_exponential_backoff_timing(self, mock_sleep, mock_openai_client):
        """Verify that sleep is called with increasing delays."""
        mock_openai_client.chat.completions.create.side_effect = [
            _make_mock_response(INVALID_JSON_RESPONSE),
            _make_mock_response(INVALID_JSON_RESPONSE),
            _make_mock_response(VALID_API_RESPONSE),
        ]
        classifier = ClinicalNoteClassifier(max_retries=3, retry_delay=2.0)
        classifier.classify_note("Test note.")
        # Attempt 1 fails -> sleep(2.0 * 1), Attempt 2 fails -> sleep(2.0 * 2)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 2.0
        assert mock_sleep.call_args_list[1][0][0] == 4.0


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestClassifyBatch:
    """Tests for ClinicalNoteClassifier.classify_batch()."""

    @pytest.fixture
    def sample_notes(self):
        return [
            ClinicalNote(
                id="note_001",
                note="72-year-old male presenting with sudden onset crushing chest pain radiating to left arm onset 40 minutes ago.",
            ),
            ClinicalNote(
                id="note_002",
                note="34-year-old female G2P1 at 32 weeks gestation presents with headache and blurred vision for 6 hours.",
            ),
        ]

    def test_batch_returns_all_results(self, mock_openai_client, sample_notes):
        mock_openai_client.chat.completions.create.return_value = _make_mock_response(
            VALID_API_RESPONSE
        )
        classifier = ClinicalNoteClassifier()
        results = classifier.classify_batch(sample_notes)
        assert len(results) == 2
        assert all(isinstance(r, ClassifiedNote) for r in results)

    def test_batch_preserves_note_ids(self, mock_openai_client, sample_notes):
        mock_openai_client.chat.completions.create.return_value = _make_mock_response(
            VALID_API_RESPONSE
        )
        classifier = ClinicalNoteClassifier()
        results = classifier.classify_batch(sample_notes)
        assert results[0].note_id == "note_001"
        assert results[1].note_id == "note_002"

    @patch("src.classifier.time.sleep")
    def test_batch_continues_on_error(self, mock_sleep, mock_openai_client, sample_notes):
        """Batch processing should continue when a note fails, using safety default."""
        # First note always fails (3 retries), second note succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            _make_mock_response(INVALID_JSON_RESPONSE),
            _make_mock_response(INVALID_JSON_RESPONSE),
            _make_mock_response(INVALID_JSON_RESPONSE),
            _make_mock_response(VALID_API_RESPONSE),
        ]
        classifier = ClinicalNoteClassifier(max_retries=3, retry_delay=0.01)
        results = classifier.classify_batch(sample_notes)
        # Both notes should have results
        assert len(results) == 2
        # First note should have error fallback
        assert results[0].classification.urgency_level == 5  # Safety default
        assert results[0].classification.primary_complaint == "CLASSIFICATION ERROR"
        assert results[0].classification.icd10_code == "R69"
        # Second note should have normal result
        assert results[1].classification.urgency_level == 3

    def test_batch_safety_default_on_error(self, mock_openai_client, sample_notes):
        """On error, batch should default to urgency 5 (EMERGENT) for safety."""
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("API down")
        classifier = ClinicalNoteClassifier()
        # classify_note raises RuntimeError, batch catches it
        results = classifier.classify_batch(sample_notes)
        for r in results:
            assert r.classification.urgency_level == 5
            assert "Manual review required" in r.classification.recommended_action

    def test_batch_progress_callback(self, mock_openai_client, sample_notes):
        mock_openai_client.chat.completions.create.return_value = _make_mock_response(
            VALID_API_RESPONSE
        )
        classifier = ClinicalNoteClassifier()
        progress_calls = []

        def on_progress(current, total, note_id):
            progress_calls.append((current, total, note_id))

        classifier.classify_batch(sample_notes, on_progress=on_progress)
        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2, "note_001")
        assert progress_calls[1] == (2, 2, "note_002")

    def test_batch_empty_list(self, mock_openai_client):
        classifier = ClinicalNoteClassifier()
        results = classifier.classify_batch([])
        assert results == []


# =============================================================================
# load_notes_from_file Tests
# =============================================================================

class TestLoadNotesFromFile:
    """Tests for loading clinical notes from JSON files."""

    def test_load_synthetic_notes(self):
        notes = load_notes_from_file(
            "data/synthetic_notes.json"
        )
        assert len(notes) == 20
        assert all(isinstance(n, ClinicalNote) for n in notes)

    def test_load_preserves_ids(self):
        notes = load_notes_from_file(
            "data/synthetic_notes.json"
        )
        ids = [n.id for n in notes]
        assert "note_001" in ids
        assert "note_020" in ids

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_notes_from_file("nonexistent_file.json")

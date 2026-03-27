"""
Tests for the evaluation pipeline (src/evaluate.py).
Covers metric computation, edge cases, and file loading.
"""

import json
import os
import tempfile

import pytest

from src.evaluate import evaluate_classifications, load_json
from src.models import EvaluationMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def perfect_match_data():
    """Actual and expected data with perfect agreement."""
    expected = [
        {"id": "note_001", "urgency_level": 5, "icd10_code": "I21.0"},
        {"id": "note_002", "urgency_level": 3, "icd10_code": "K35.80"},
        {"id": "note_003", "urgency_level": 1, "icd10_code": "Z00.00"},
    ]
    actual = [
        {"note_id": "note_001", "classification": {"urgency_level": 5, "icd10_code": "I21.0"}},
        {"note_id": "note_002", "classification": {"urgency_level": 3, "icd10_code": "K35.80"}},
        {"note_id": "note_003", "classification": {"urgency_level": 1, "icd10_code": "Z00.00"}},
    ]
    return actual, expected


@pytest.fixture
def partial_match_data():
    """Actual and expected data with partial agreement."""
    expected = [
        {"id": "note_001", "urgency_level": 5, "icd10_code": "I21.0"},
        {"id": "note_002", "urgency_level": 3, "icd10_code": "K35.80"},
        {"id": "note_003", "urgency_level": 1, "icd10_code": "Z00.00"},
    ]
    actual = [
        {"note_id": "note_001", "classification": {"urgency_level": 5, "icd10_code": "I21.0"}},
        {"note_id": "note_002", "classification": {"urgency_level": 4, "icd10_code": "K35.80"}},
        {"note_id": "note_003", "classification": {"urgency_level": 1, "icd10_code": "Z00.129"}},
    ]
    return actual, expected


# =============================================================================
# evaluate_classifications Tests
# =============================================================================

class TestEvaluateClassifications:
    """Tests for the core evaluation function."""

    def test_perfect_match(self, perfect_match_data):
        actual, expected = perfect_match_data
        metrics, detailed = evaluate_classifications(actual, expected)
        assert metrics.total_notes == 3
        assert metrics.urgency_exact_match == 3
        assert metrics.urgency_within_one == 3
        assert metrics.urgency_mae == 0.0
        assert metrics.icd10_match_rate == 1.0

    def test_partial_match_urgency(self, partial_match_data):
        actual, expected = partial_match_data
        metrics, detailed = evaluate_classifications(actual, expected)
        assert metrics.total_notes == 3
        # note_001: exact, note_002: off by 1, note_003: exact
        assert metrics.urgency_exact_match == 2
        assert metrics.urgency_within_one == 3
        assert metrics.urgency_mae == pytest.approx(1 / 3, abs=0.01)

    def test_partial_match_icd10(self, partial_match_data):
        actual, expected = partial_match_data
        metrics, detailed = evaluate_classifications(actual, expected)
        # note_001: match, note_002: match, note_003: mismatch
        assert metrics.icd10_match_rate == pytest.approx(2 / 3, abs=0.01)

    def test_detailed_results_structure(self, perfect_match_data):
        actual, expected = perfect_match_data
        metrics, detailed = evaluate_classifications(actual, expected)
        assert len(detailed) == 3
        for d in detailed:
            assert "note_id" in d
            assert "urgency_actual" in d
            assert "urgency_expected" in d
            assert "urgency_diff" in d
            assert "icd10_actual" in d
            assert "icd10_expected" in d
            assert "icd10_match" in d

    def test_missing_note_id_skipped(self):
        expected = [{"id": "note_001", "urgency_level": 5, "icd10_code": "I21.0"}]
        actual = [
            {"note_id": "note_001", "classification": {"urgency_level": 5, "icd10_code": "I21.0"}},
            {"note_id": "note_999", "classification": {"urgency_level": 3, "icd10_code": "K35.80"}},
        ]
        metrics, detailed = evaluate_classifications(actual, expected)
        assert metrics.total_notes == 1
        assert len(detailed) == 1

    def test_empty_inputs(self):
        metrics, detailed = evaluate_classifications([], [])
        assert metrics.total_notes == 0
        assert metrics.urgency_mae == 0.0
        assert metrics.icd10_match_rate == 0.0
        assert detailed == []

    def test_large_urgency_difference(self):
        expected = [{"id": "note_001", "urgency_level": 1, "icd10_code": "Z00.00"}]
        actual = [
            {"note_id": "note_001", "classification": {"urgency_level": 5, "icd10_code": "Z00.00"}},
        ]
        metrics, detailed = evaluate_classifications(actual, expected)
        assert metrics.urgency_exact_match == 0
        assert metrics.urgency_within_one == 0
        assert metrics.urgency_mae == 4.0

    def test_handles_id_key_in_actual(self):
        """Evaluate should handle 'id' key as well as 'note_id' in actual results."""
        expected = [{"id": "note_001", "urgency_level": 5, "icd10_code": "I21.0"}]
        actual = [
            {"id": "note_001", "urgency_level": 5, "icd10_code": "I21.0"},
        ]
        metrics, detailed = evaluate_classifications(actual, expected)
        assert metrics.total_notes == 1
        assert metrics.urgency_exact_match == 1


# =============================================================================
# load_json Tests
# =============================================================================

class TestLoadJson:
    """Tests for JSON file loading."""

    def test_load_valid_json(self):
        data = [{"id": "test", "value": 42}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name
        try:
            loaded = load_json(tmp_path)
            assert loaded == data
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_json("/nonexistent/path/file.json")

    def test_load_expected_outputs_file(self):
        data = load_json("data/expected_outputs.json")
        assert len(data) == 20
        assert data[0]["id"] == "note_001"

    def test_load_sample_classification_file(self):
        data = load_json("outputs/sample_classification.json")
        assert len(data) == 20
        assert data[0]["note_id"] == "note_001"

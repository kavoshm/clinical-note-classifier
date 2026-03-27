"""
Tests for Pydantic data models (src/models.py).
Covers validation rules, field constraints, edge cases, and serialization.
"""

import pytest
from pydantic import ValidationError

from src.models import (
    ClassificationResult,
    ClassifiedNote,
    ClinicalNote,
    EvaluationMetrics,
    UrgencyLevel,
)


# =============================================================================
# UrgencyLevel Enum Tests
# =============================================================================

class TestUrgencyLevel:
    """Tests for the UrgencyLevel IntEnum."""

    def test_urgency_values(self):
        assert UrgencyLevel.ROUTINE == 1
        assert UrgencyLevel.LOW == 2
        assert UrgencyLevel.MODERATE == 3
        assert UrgencyLevel.HIGH == 4
        assert UrgencyLevel.EMERGENT == 5

    def test_urgency_is_int(self):
        assert isinstance(UrgencyLevel.ROUTINE, int)
        assert UrgencyLevel.EMERGENT + 0 == 5


# =============================================================================
# ClassificationResult Tests
# =============================================================================

class TestClassificationResult:
    """Tests for ClassificationResult Pydantic model."""

    @pytest.fixture
    def valid_result_data(self):
        return {
            "urgency_level": 3,
            "primary_complaint": "Acute appendicitis",
            "icd10_code": "K35.80",
            "icd10_description": "Unspecified acute appendicitis without abscess",
            "reasoning": "Patient presents with classic signs of acute appendicitis including RLQ pain and elevated WBC.",
            "recommended_action": "Surgical consult for appendectomy, NPO status, IV fluids",
        }

    def test_valid_classification(self, valid_result_data):
        result = ClassificationResult(**valid_result_data)
        assert result.urgency_level == 3
        assert result.primary_complaint == "Acute appendicitis"
        assert result.icd10_code == "K35.80"

    # -- Urgency level validation --

    def test_urgency_level_minimum(self, valid_result_data):
        valid_result_data["urgency_level"] = 1
        result = ClassificationResult(**valid_result_data)
        assert result.urgency_level == 1

    def test_urgency_level_maximum(self, valid_result_data):
        valid_result_data["urgency_level"] = 5
        result = ClassificationResult(**valid_result_data)
        assert result.urgency_level == 5

    def test_urgency_level_below_minimum(self, valid_result_data):
        valid_result_data["urgency_level"] = 0
        with pytest.raises(ValidationError) as exc_info:
            ClassificationResult(**valid_result_data)
        assert "urgency_level" in str(exc_info.value)

    def test_urgency_level_above_maximum(self, valid_result_data):
        valid_result_data["urgency_level"] = 6
        with pytest.raises(ValidationError) as exc_info:
            ClassificationResult(**valid_result_data)
        assert "urgency_level" in str(exc_info.value)

    def test_urgency_level_negative(self, valid_result_data):
        valid_result_data["urgency_level"] = -1
        with pytest.raises(ValidationError):
            ClassificationResult(**valid_result_data)

    def test_urgency_level_non_integer_string(self, valid_result_data):
        valid_result_data["urgency_level"] = "high"
        with pytest.raises(ValidationError):
            ClassificationResult(**valid_result_data)

    # -- ICD-10 code validation --

    def test_icd10_code_valid_formats(self, valid_result_data):
        valid_codes = ["I21.0", "E11.65", "Z00.129", "T07", "R69", "F32.2", "A15.0"]
        for code in valid_codes:
            valid_result_data["icd10_code"] = code
            result = ClassificationResult(**valid_result_data)
            assert result.icd10_code == code

    def test_icd10_code_too_short(self, valid_result_data):
        valid_result_data["icd10_code"] = "I2"
        with pytest.raises(ValidationError, match="too short"):
            ClassificationResult(**valid_result_data)

    def test_icd10_code_must_start_with_letter(self, valid_result_data):
        valid_result_data["icd10_code"] = "123.4"
        with pytest.raises(ValidationError, match="must start with a letter"):
            ClassificationResult(**valid_result_data)

    def test_icd10_code_stripped_of_whitespace(self, valid_result_data):
        valid_result_data["icd10_code"] = "  I21.0  "
        result = ClassificationResult(**valid_result_data)
        assert result.icd10_code == "I21.0"

    def test_icd10_code_empty_string(self, valid_result_data):
        valid_result_data["icd10_code"] = ""
        with pytest.raises(ValidationError):
            ClassificationResult(**valid_result_data)

    # -- String field min_length constraints --

    def test_primary_complaint_too_short(self, valid_result_data):
        valid_result_data["primary_complaint"] = "AB"
        with pytest.raises(ValidationError):
            ClassificationResult(**valid_result_data)

    def test_reasoning_too_short(self, valid_result_data):
        valid_result_data["reasoning"] = "Short."
        with pytest.raises(ValidationError):
            ClassificationResult(**valid_result_data)

    def test_recommended_action_too_short(self, valid_result_data):
        valid_result_data["recommended_action"] = "Do thing"
        with pytest.raises(ValidationError):
            ClassificationResult(**valid_result_data)

    # -- Missing required fields --

    def test_missing_urgency_level(self, valid_result_data):
        del valid_result_data["urgency_level"]
        with pytest.raises(ValidationError):
            ClassificationResult(**valid_result_data)

    def test_missing_icd10_code(self, valid_result_data):
        del valid_result_data["icd10_code"]
        with pytest.raises(ValidationError):
            ClassificationResult(**valid_result_data)

    def test_missing_reasoning(self, valid_result_data):
        del valid_result_data["reasoning"]
        with pytest.raises(ValidationError):
            ClassificationResult(**valid_result_data)

    # -- Properties and serialization --

    def test_urgency_label_property(self, valid_result_data):
        labels = {1: "ROUTINE", 2: "LOW", 3: "MODERATE", 4: "HIGH", 5: "EMERGENT"}
        for level, label in labels.items():
            valid_result_data["urgency_level"] = level
            result = ClassificationResult(**valid_result_data)
            assert result.urgency_label == label

    def test_urgency_label_unknown_level(self):
        """Test that urgency_label returns UNKNOWN for out-of-range levels.

        Note: This tests the property behavior. In practice, Pydantic
        would reject levels outside 1-5 during validation.
        """
        # The property handles arbitrary int values gracefully
        result = ClassificationResult.model_construct(urgency_level=99)
        assert result.urgency_label == "UNKNOWN"

    def test_to_flat_dict(self, valid_result_data):
        result = ClassificationResult(**valid_result_data)
        flat = result.to_flat_dict()
        assert flat["urgency_level"] == 3
        assert flat["urgency_label"] == "MODERATE"
        assert flat["primary_complaint"] == "Acute appendicitis"
        assert flat["icd10_code"] == "K35.80"
        assert "reasoning" in flat
        assert "recommended_action" in flat

    def test_model_dump_roundtrip(self, valid_result_data):
        result = ClassificationResult(**valid_result_data)
        dumped = result.model_dump()
        reconstructed = ClassificationResult(**dumped)
        assert reconstructed.urgency_level == result.urgency_level
        assert reconstructed.icd10_code == result.icd10_code


# =============================================================================
# ClinicalNote Tests
# =============================================================================

class TestClinicalNote:
    """Tests for ClinicalNote Pydantic model."""

    @pytest.fixture
    def valid_note_data(self):
        return {
            "id": "note_001",
            "note": "72-year-old male presenting to ED with sudden onset crushing substernal chest pain radiating to left arm.",
            "category": "cardiac_emergency",
        }

    def test_valid_clinical_note(self, valid_note_data):
        note = ClinicalNote(**valid_note_data)
        assert note.id == "note_001"
        assert len(note.note) >= 50

    def test_note_too_short(self, valid_note_data):
        valid_note_data["note"] = "Short note text."
        with pytest.raises(ValidationError):
            ClinicalNote(**valid_note_data)

    def test_note_exactly_50_characters(self):
        note_text = "A" * 50
        note = ClinicalNote(id="test", note=note_text)
        assert len(note.note) == 50

    def test_category_is_optional(self, valid_note_data):
        del valid_note_data["category"]
        note = ClinicalNote(**valid_note_data)
        assert note.category is None

    def test_missing_id(self, valid_note_data):
        del valid_note_data["id"]
        with pytest.raises(ValidationError):
            ClinicalNote(**valid_note_data)

    def test_missing_note_text(self, valid_note_data):
        del valid_note_data["note"]
        with pytest.raises(ValidationError):
            ClinicalNote(**valid_note_data)


# =============================================================================
# ClassifiedNote Tests
# =============================================================================

class TestClassifiedNote:
    """Tests for ClassifiedNote Pydantic model."""

    @pytest.fixture
    def valid_classified_data(self):
        return {
            "note_id": "note_001",
            "note_text": "Sample clinical note text that is long enough for testing purposes.",
            "classification": ClassificationResult(
                urgency_level=5,
                primary_complaint="Acute STEMI",
                icd10_code="I21.0",
                icd10_description="ST elevation MI",
                reasoning="Classic STEMI presentation with ST elevation and elevated troponin.",
                recommended_action="Activate cath lab STAT, dual antiplatelet therapy",
            ),
        }

    def test_valid_classified_note(self, valid_classified_data):
        classified = ClassifiedNote(**valid_classified_data)
        assert classified.note_id == "note_001"
        assert classified.classification.urgency_level == 5

    def test_default_model_used(self, valid_classified_data):
        classified = ClassifiedNote(**valid_classified_data)
        assert classified.model_used == "gpt-4o-mini"

    def test_default_prompt_version(self, valid_classified_data):
        classified = ClassifiedNote(**valid_classified_data)
        assert classified.prompt_version == "v1.0"

    def test_custom_model_and_version(self, valid_classified_data):
        valid_classified_data["model_used"] = "gpt-4o"
        valid_classified_data["prompt_version"] = "v2.0"
        classified = ClassifiedNote(**valid_classified_data)
        assert classified.model_used == "gpt-4o"
        assert classified.prompt_version == "v2.0"

    def test_to_summary_dict(self, valid_classified_data):
        classified = ClassifiedNote(**valid_classified_data)
        summary = classified.to_summary_dict()
        assert summary["note_id"] == "note_001"
        assert summary["urgency_level"] == 5
        assert summary["urgency_label"] == "EMERGENT"
        assert summary["icd10_code"] == "I21.0"
        assert summary["model"] == "gpt-4o-mini"
        assert summary["prompt_version"] == "v1.0"


# =============================================================================
# EvaluationMetrics Tests
# =============================================================================

class TestEvaluationMetrics:
    """Tests for EvaluationMetrics Pydantic model."""

    @pytest.fixture
    def valid_metrics_data(self):
        return {
            "total_notes": 20,
            "urgency_exact_match": 18,
            "urgency_within_one": 20,
            "urgency_mae": 0.1,
            "icd10_match_rate": 0.85,
            "notes_evaluated": [f"note_{i:03d}" for i in range(1, 21)],
        }

    def test_valid_metrics(self, valid_metrics_data):
        metrics = EvaluationMetrics(**valid_metrics_data)
        assert metrics.total_notes == 20
        assert metrics.urgency_exact_match == 18

    def test_urgency_exact_accuracy(self, valid_metrics_data):
        metrics = EvaluationMetrics(**valid_metrics_data)
        assert metrics.urgency_exact_accuracy == 18 / 20

    def test_urgency_within_one_accuracy(self, valid_metrics_data):
        metrics = EvaluationMetrics(**valid_metrics_data)
        assert metrics.urgency_within_one_accuracy == 1.0

    def test_zero_total_notes_accuracy(self):
        metrics = EvaluationMetrics(
            total_notes=0,
            urgency_exact_match=0,
            urgency_within_one=0,
            urgency_mae=0.0,
            icd10_match_rate=0.0,
            notes_evaluated=[],
        )
        assert metrics.urgency_exact_accuracy == 0.0
        assert metrics.urgency_within_one_accuracy == 0.0

"""
Tests for CSV and JSON export functionality.
Covers the serialization paths used by batch classification output.
"""

import csv
import json
import io
import tempfile
import os

import pytest

from src.models import ClassificationResult, ClassifiedNote


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_classified_notes():
    """Create a list of ClassifiedNote objects for export testing."""
    results = []
    test_data = [
        ("note_001", 5, "Acute STEMI", "I21.0", "ST elevation MI involving left main coronary artery"),
        ("note_002", 3, "Acute appendicitis", "K35.80", "Unspecified acute appendicitis"),
        ("note_003", 1, "Annual wellness exam", "Z00.00", "General adult medical exam"),
    ]
    for note_id, urgency, complaint, code, desc in test_data:
        classification = ClassificationResult(
            urgency_level=urgency,
            primary_complaint=complaint,
            icd10_code=code,
            icd10_description=desc,
            reasoning=f"Clinical reasoning for {complaint} with supporting evidence from the note.",
            recommended_action=f"Recommended clinical action plan for {complaint} management.",
        )
        classified = ClassifiedNote(
            note_id=note_id,
            note_text=f"Sample clinical note text for {note_id} that is sufficiently long for testing.",
            classification=classification,
            model_used="gpt-4o-mini",
            prompt_version="v1.0",
        )
        results.append(classified)
    return results


# =============================================================================
# JSON Export Tests
# =============================================================================

class TestJsonExport:
    """Tests for JSON export format matching batch output."""

    def test_json_export_structure(self, sample_classified_notes):
        """JSON export should match the structure used in main.py classify_batch."""
        json_data = []
        for r in sample_classified_notes:
            entry = {
                "note_id": r.note_id,
                "classification": r.classification.model_dump(),
            }
            entry["classification"]["urgency_label"] = r.classification.urgency_label
            json_data.append(entry)

        assert len(json_data) == 3
        assert json_data[0]["note_id"] == "note_001"
        assert json_data[0]["classification"]["urgency_level"] == 5
        assert json_data[0]["classification"]["urgency_label"] == "EMERGENT"

    def test_json_serializable(self, sample_classified_notes):
        """Exported JSON data should be serializable without errors."""
        json_data = []
        for r in sample_classified_notes:
            entry = {
                "note_id": r.note_id,
                "classification": r.classification.model_dump(),
            }
            entry["classification"]["urgency_label"] = r.classification.urgency_label
            json_data.append(entry)

        serialized = json.dumps(json_data, indent=2)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 3
        assert deserialized[0]["note_id"] == "note_001"

    def test_json_write_and_read_roundtrip(self, sample_classified_notes):
        """JSON should survive write-to-file and read-from-file."""
        json_data = []
        for r in sample_classified_notes:
            entry = {
                "note_id": r.note_id,
                "classification": r.classification.model_dump(),
            }
            json_data.append(entry)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f, indent=2)
            tmp_path = f.name

        try:
            with open(tmp_path, "r") as f:
                loaded = json.load(f)
            assert len(loaded) == 3
            assert loaded[0]["note_id"] == "note_001"
            assert loaded[0]["classification"]["urgency_level"] == 5
        finally:
            os.unlink(tmp_path)


# =============================================================================
# CSV Export Tests
# =============================================================================

class TestCsvExport:
    """Tests for CSV export format matching batch output."""

    def test_csv_export_headers(self, sample_classified_notes):
        """CSV export should have the correct column headers."""
        fieldnames = list(sample_classified_notes[0].to_summary_dict().keys())
        expected_fields = [
            "note_id", "urgency_level", "urgency_label", "primary_complaint",
            "icd10_code", "icd10_description", "recommended_action", "model",
            "prompt_version",
        ]
        assert fieldnames == expected_fields

    def test_csv_export_row_count(self, sample_classified_notes):
        """CSV should have one row per classified note plus header."""
        output = io.StringIO()
        fieldnames = list(sample_classified_notes[0].to_summary_dict().keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample_classified_notes:
            writer.writerow(r.to_summary_dict())

        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)
        assert len(rows) == 3

    def test_csv_data_integrity(self, sample_classified_notes):
        """CSV data values should match the source objects."""
        output = io.StringIO()
        fieldnames = list(sample_classified_notes[0].to_summary_dict().keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample_classified_notes:
            writer.writerow(r.to_summary_dict())

        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)

        assert rows[0]["note_id"] == "note_001"
        assert rows[0]["urgency_level"] == "5"  # CSV stores as string
        assert rows[0]["urgency_label"] == "EMERGENT"
        assert rows[0]["icd10_code"] == "I21.0"
        assert rows[0]["model"] == "gpt-4o-mini"

    def test_csv_write_and_read_roundtrip(self, sample_classified_notes):
        """CSV should survive write-to-file and read-from-file."""
        fieldnames = list(sample_classified_notes[0].to_summary_dict().keys())

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sample_classified_notes:
                writer.writerow(r.to_summary_dict())
            tmp_path = f.name

        try:
            with open(tmp_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 3
            assert rows[2]["note_id"] == "note_003"
            assert rows[2]["urgency_label"] == "ROUTINE"
        finally:
            os.unlink(tmp_path)

    def test_csv_handles_commas_in_fields(self, sample_classified_notes):
        """CSV export should properly handle commas in text fields."""
        # Modify a complaint to include commas
        sample_classified_notes[0].classification = ClassificationResult(
            urgency_level=5,
            primary_complaint="Acute STEMI, with cardiogenic shock, hemodynamic instability",
            icd10_code="I21.0",
            icd10_description="ST elevation MI, left main coronary artery",
            reasoning="Patient presents with classic STEMI findings, including ST elevation and troponin.",
            recommended_action="Activate cath lab, administer dual antiplatelet, IV heparin",
        )

        output = io.StringIO()
        fieldnames = list(sample_classified_notes[0].to_summary_dict().keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample_classified_notes:
            writer.writerow(r.to_summary_dict())

        output.seek(0)
        reader = csv.DictReader(output)
        rows = list(reader)
        assert "cardiogenic shock" in rows[0]["primary_complaint"]

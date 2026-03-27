"""
Pydantic Models for Clinical Note Classification
==================================================
Defines the structured data models used throughout the classifier pipeline.
Pydantic provides runtime validation, serialization, and clear documentation
of the expected data shapes — critical for healthcare AI where data integrity
is non-negotiable.
"""

from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class UrgencyLevel(IntEnum):
    """Clinical urgency scale aligned with ESI (Emergency Severity Index)."""
    ROUTINE = 1          # Preventive care, stable follow-ups
    LOW = 2              # Minor acute issues, can wait 24-48 hours
    MODERATE = 3         # Needs evaluation within hours
    HIGH = 4             # Needs prompt intervention, risk of deterioration
    EMERGENT = 5         # Life-threatening, immediate intervention required


class ClassificationResult(BaseModel):
    """Structured output from the clinical note classifier."""

    urgency_level: int = Field(
        ...,
        ge=1,
        le=5,
        description="Urgency level 1-5 (1=routine, 5=emergent)",
    )
    primary_complaint: str = Field(
        ...,
        min_length=3,
        description="Primary clinical complaint or diagnosis",
    )
    icd10_code: str = Field(
        ...,
        description="Most relevant ICD-10-CM code (e.g., 'I21.0')",
    )
    icd10_description: str = Field(
        ...,
        description="Human-readable description of the ICD-10 code",
    )
    reasoning: str = Field(
        ...,
        min_length=20,
        description="Clinical reasoning supporting the classification",
    )
    recommended_action: str = Field(
        ...,
        min_length=10,
        description="Recommended next steps or disposition",
    )

    @field_validator("icd10_code")
    @classmethod
    def validate_icd10_format(cls, v: str) -> str:
        """Validate ICD-10 code follows the expected format pattern."""
        # ICD-10-CM codes: letter followed by digits, optional decimal and more digits
        # Examples: I21.0, E11.65, Z00.129, T07
        stripped = v.strip()
        if len(stripped) < 3:
            raise ValueError(f"ICD-10 code '{stripped}' is too short (minimum 3 characters)")
        if not stripped[0].isalpha():
            raise ValueError(f"ICD-10 code must start with a letter, got '{stripped[0]}'")
        return stripped

    @property
    def urgency_label(self) -> str:
        """Return human-readable urgency label."""
        labels = {
            1: "ROUTINE",
            2: "LOW",
            3: "MODERATE",
            4: "HIGH",
            5: "EMERGENT",
        }
        return labels.get(self.urgency_level, "UNKNOWN")

    def to_flat_dict(self) -> dict:
        """Return a flat dictionary suitable for CSV export."""
        return {
            "urgency_level": self.urgency_level,
            "urgency_label": self.urgency_label,
            "primary_complaint": self.primary_complaint,
            "icd10_code": self.icd10_code,
            "icd10_description": self.icd10_description,
            "reasoning": self.reasoning,
            "recommended_action": self.recommended_action,
        }


class ClinicalNote(BaseModel):
    """Input model representing a clinical note to be classified."""

    id: str = Field(..., description="Unique identifier for the note")
    note: str = Field(
        ...,
        min_length=50,
        description="Free-text clinical note content",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional ground-truth category for evaluation",
    )


class ClassifiedNote(BaseModel):
    """A clinical note paired with its classification result."""

    note_id: str
    note_text: str
    classification: ClassificationResult
    model_used: str = Field(default="gpt-4o-mini", description="Model used for classification")
    prompt_version: str = Field(default="v1.0", description="Version of the prompt used")

    def to_summary_dict(self) -> dict:
        """Return a summary dictionary for batch result reporting."""
        return {
            "note_id": self.note_id,
            "urgency_level": self.classification.urgency_level,
            "urgency_label": self.classification.urgency_label,
            "primary_complaint": self.classification.primary_complaint,
            "icd10_code": self.classification.icd10_code,
            "icd10_description": self.classification.icd10_description,
            "recommended_action": self.classification.recommended_action,
            "model": self.model_used,
            "prompt_version": self.prompt_version,
        }


class EvaluationMetrics(BaseModel):
    """Metrics from comparing classifier output against expected results."""

    total_notes: int
    urgency_exact_match: int
    urgency_within_one: int
    urgency_mae: float = Field(description="Mean Absolute Error for urgency levels")
    icd10_match_rate: float = Field(description="Rate of exact ICD-10 code match")
    notes_evaluated: list[str] = Field(description="List of note IDs evaluated")

    @property
    def urgency_exact_accuracy(self) -> float:
        """Exact match accuracy for urgency levels."""
        if self.total_notes == 0:
            return 0.0
        return self.urgency_exact_match / self.total_notes

    @property
    def urgency_within_one_accuracy(self) -> float:
        """Accuracy when allowing +/- 1 tolerance on urgency."""
        if self.total_notes == 0:
            return 0.0
        return self.urgency_within_one / self.total_notes

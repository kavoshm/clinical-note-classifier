"""
Input Validation for Clinical Note Content
============================================
Validates that input text is appropriate for clinical classification.
Goes beyond basic length checks to detect non-clinical content,
prompt injection attempts, and structurally invalid inputs.

This module provides defense-in-depth: even though Pydantic validates
the data model (ClinicalNote.min_length=50), this validates the
*content* is actually clinical in nature and safe to process.
"""

import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of clinical content validation."""
    is_valid: bool
    reason: str


# Minimum character length for a meaningful clinical note
MIN_CLINICAL_LENGTH = 50

# Clinical indicator terms — presence of several suggests clinical content.
# Weighted toward terms unlikely to appear in non-clinical text.
CLINICAL_INDICATORS = [
    # Demographics / history
    r"\b\d{1,3}[- ]?year[- ]?old\b",
    r"\bpatient\b",
    r"\bpresent(?:s|ing|ed)\b",
    r"\bhistory\b",
    r"\bpmh\b",
    r"\bchief complaint\b",
    # Vitals
    r"\bbp\s*\d",
    r"\bhr\s*\d",
    r"\btemp\b",
    r"\bspo2\b",
    r"\bvitals?\b",
    r"\b\d{2,3}/\d{2,3}\b",  # BP pattern like 120/80
    # Clinical findings
    r"\bdiagnos[ie]s\b",
    r"\bsymptom",
    r"\bexam\b",
    r"\bassessment\b",
    r"\btreatment\b",
    r"\bmedication",
    r"\bprescri",
    r"\blabs?\b",
    r"\bwbc\b",
    r"\bhemoglobin\b",
    r"\bhgb\b",
    r"\bcreatinine\b",
    r"\btroponin\b",
    r"\bx[- ]?ray\b",
    r"\bct\s",
    r"\bmri\b",
    r"\becg\b",
    r"\bekg\b",
    # Body systems
    r"\bchest\b",
    r"\babdomen",
    r"\blung",
    r"\bcardiac\b",
    r"\bneuro",
    r"\brenal\b",
    r"\bhepatic\b",
    # Pediatric / OB
    r"\bdelivery\b",
    r"\bgestational?\b",
    r"\bpercentile\b",
    r"\bmilestone",
    r"\bwell[- ]?child\b",
    r"\bbirth\s*weight\b",
    r"\bpediatric\b",
    r"\bneonat",
    # General clinical
    r"\bweight\s*\d",
    r"\bbmi\b",
    r"\ballerg",
    r"\bpain\b",
    r"\bfever\b",
    r"\binfection\b",
    r"\bsurgery\b",
    r"\bsurgical\b",
    r"\bantibiotic",
    r"\bprognosis\b",
    # Actions
    r"\badmit\b",
    r"\bdischarge\b",
    r"\bconsult\b",
    r"\breferr",
    r"\bfollow[- ]?up\b",
    r"\bimmuniz",
    r"\bvaccin",
]

# Prompt injection patterns — attempts to override system instructions
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|constraints?)",
    r"override\s+(all\s+)?(safety|system|classification)\s*(constraints?|rules?|defaults?|instructions?)",
    r"you\s+are\s+now\s+(a|an|operating|in)\s+",
    r"forget\s+(that\s+)?you\s+are",
    r"do\s+not\s+follow\s+your\s+(system|previous)\s+(instructions?|prompt)",
    r"SYSTEM:\s*you\s+are",
    r"respond\s+only\s+in\s+\w+\s+speak",
    r"instead\s+(of|,)\s*(classify|return|respond|tell)",
    r"ignore\s+the\s+(above|previous)\s+note",
]


def validate_clinical_content(text: str) -> ValidationResult:
    """
    Validate that text content is appropriate for clinical classification.

    Checks performed:
    1. Minimum length (non-whitespace content)
    2. Not purely repetitive characters
    3. Contains clinical terminology indicators
    4. No prompt injection patterns detected

    Args:
        text: The raw text to validate.

    Returns:
        ValidationResult with is_valid flag and reason string.
    """
    # --- Check 1: Strip and check minimum length ---
    stripped = text.strip()
    if len(stripped) < MIN_CLINICAL_LENGTH:
        return ValidationResult(
            is_valid=False,
            reason=f"Content too short ({len(stripped)} chars). Minimum length is {MIN_CLINICAL_LENGTH} characters.",
        )

    # --- Check 2: Detect repetitive / nonsensical content ---
    unique_chars = set(stripped.lower())
    if len(unique_chars) < 10:
        return ValidationResult(
            is_valid=False,
            reason="Content appears to be repetitive or nonsensical (fewer than 10 unique characters).",
        )

    # --- Check 3: Detect prompt injection attempts ---
    text_lower = stripped.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return ValidationResult(
                is_valid=False,
                reason="Potential prompt injection detected. Input contains instruction-override patterns.",
            )

    # --- Check 4: Check for clinical content indicators ---
    match_count = 0
    for pattern in CLINICAL_INDICATORS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            match_count += 1
        if match_count >= 3:
            break  # Enough evidence of clinical content

    if match_count < 3:
        return ValidationResult(
            is_valid=False,
            reason=(
                f"Content does not appear to be clinical in nature "
                f"(matched {match_count}/3 required clinical indicators). "
                f"Expected medical terminology such as patient demographics, "
                f"vitals, symptoms, diagnoses, or treatment plans."
            ),
        )

    return ValidationResult(is_valid=True, reason="Content passed all validation checks.")

"""
Clinical Note Classifier — Core Engine
========================================
Main classifier that takes free-text clinical notes and returns structured
classifications using OpenAI's API. Supports single-note and batch
classification with retry logic, error handling, and structured output
validation via Pydantic.

This is the heart of the 1-3 project, combining prompt engineering patterns
from 1-1 and system building patterns from 1-2.
"""

import json
import time
from typing import Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from pydantic import ValidationError

from src.logging_config import get_logger
from src.models import ClassificationResult, ClassifiedNote, ClinicalNote
from src.prompts import get_system_prompt, get_few_shot_examples, build_user_message

logger = get_logger(__name__)


class ClinicalNoteClassifier:
    """Classifies clinical notes using OpenAI LLMs with structured output."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        prompt_version: str = "v1.0",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        """
        Initialize the classifier.

        Args:
            model: OpenAI model to use for classification.
            prompt_version: Version of the prompt to use (for version tracking).
            temperature: Sampling temperature. Use 0 for deterministic output.
            max_retries: Maximum number of retry attempts on failure.
            retry_delay: Base delay between retries in seconds (exponential backoff).
        """
        self.client = OpenAI()
        self.model = model
        self.prompt_version = prompt_version
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Build the full system message (prompt + few-shot examples)
        self._system_message = (
            get_system_prompt(prompt_version) + "\n\n" + get_few_shot_examples(prompt_version)
        )

        logger.info(
            "Classifier initialized",
            extra={
                "model": self.model,
                "prompt_version": self.prompt_version,
                "temperature": self.temperature,
                "max_retries": self.max_retries,
            },
        )

    def classify_note(self, note_text: str) -> ClassificationResult:
        """
        Classify a single clinical note.

        Args:
            note_text: Free-text clinical note content.

        Returns:
            ClassificationResult with urgency, complaint, ICD-10, reasoning, action.

        Raises:
            RuntimeError: If classification fails after all retries.
        """
        user_message = build_user_message(note_text, self.prompt_version)
        logger.info(
            "Classification request started",
            extra={"model": self.model, "note_length": len(note_text)},
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self._system_message},
                        {"role": "user", "content": user_message},
                    ],
                )

                raw_content = response.choices[0].message.content
                parsed = json.loads(raw_content)
                result = ClassificationResult(**parsed)
                logger.info(
                    "Classification succeeded",
                    extra={
                        "urgency_level": result.urgency_level,
                        "urgency_label": result.urgency_label,
                        "icd10_code": result.icd10_code,
                        "attempt": attempt,
                        "model": self.model,
                    },
                )
                return result

            except (json.JSONDecodeError, ValidationError) as e:
                # Response was returned but didn't parse correctly
                logger.warning(
                    "Classification parse/validation error",
                    extra={"attempt": attempt, "error_type": type(e).__name__, "error": str(e)},
                )
                if attempt == self.max_retries:
                    logger.error(
                        "Classification failed after max retries (parse/validation)",
                        extra={"max_retries": self.max_retries, "error_type": type(e).__name__},
                    )
                    raise RuntimeError(
                        f"Classification failed after {self.max_retries} attempts. "
                        f"Last error: {type(e).__name__}: {e}"
                    )
                time.sleep(self.retry_delay * attempt)

            except RateLimitError:
                # Rate limited — wait and retry with exponential backoff
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "Rate limited by API",
                    extra={"attempt": attempt, "wait_time": wait_time},
                )
                if attempt == self.max_retries:
                    logger.error("Rate limited after max retries", extra={"max_retries": self.max_retries})
                    raise RuntimeError(
                        f"Rate limited after {self.max_retries} attempts."
                    )
                time.sleep(wait_time)

            except APITimeoutError:
                # API timeout — retry with backoff
                logger.warning("API timeout", extra={"attempt": attempt})
                if attempt == self.max_retries:
                    logger.error("API timeout after max retries", extra={"max_retries": self.max_retries})
                    raise RuntimeError(
                        f"API timeout after {self.max_retries} attempts."
                    )
                time.sleep(self.retry_delay * attempt)

            except APIError as e:
                # Other API errors
                logger.warning("API error", extra={"attempt": attempt, "error": str(e)})
                if attempt == self.max_retries:
                    logger.error("API error after max retries", extra={"max_retries": self.max_retries, "error": str(e)})
                    raise RuntimeError(
                        f"API error after {self.max_retries} attempts: {e}"
                    )
                time.sleep(self.retry_delay * attempt)

        # Should not reach here, but just in case
        raise RuntimeError("Classification failed unexpectedly.")

    def classify_batch(
        self,
        notes: list[ClinicalNote],
        on_progress: Optional[callable] = None,
    ) -> list[ClassifiedNote]:
        """
        Classify a batch of clinical notes.

        Args:
            notes: List of ClinicalNote objects to classify.
            on_progress: Optional callback(current, total, note_id) for progress tracking.

        Returns:
            List of ClassifiedNote objects with classification results.
        """
        results: list[ClassifiedNote] = []
        total = len(notes)
        logger.info("Batch classification started", extra={"total_notes": total, "model": self.model})

        for i, note in enumerate(notes):
            if on_progress:
                on_progress(i + 1, total, note.id)

            try:
                classification = self.classify_note(note.note)
                classified = ClassifiedNote(
                    note_id=note.id,
                    note_text=note.note,
                    classification=classification,
                    model_used=self.model,
                    prompt_version=self.prompt_version,
                )
                results.append(classified)
            except RuntimeError as e:
                logger.error(
                    "Batch note classification failed, using safety default",
                    extra={"note_id": note.id, "error": str(e), "safety_urgency": 5},
                )
                # Log the error but continue with remaining notes
                error_classification = ClassificationResult(
                    urgency_level=5,  # Default to highest urgency on error (safe default)
                    primary_complaint="CLASSIFICATION ERROR",
                    icd10_code="R69",
                    icd10_description="Illness, unspecified",
                    reasoning=f"Automated classification failed: {str(e)}. Defaulting to highest urgency for safety.",
                    recommended_action="Manual review required — automated classification was unable to process this note.",
                )
                classified = ClassifiedNote(
                    note_id=note.id,
                    note_text=note.note,
                    classification=error_classification,
                    model_used=self.model,
                    prompt_version=self.prompt_version,
                )
                results.append(classified)

        logger.info(
            "Batch classification completed",
            extra={"total_notes": total, "successful": len(results), "model": self.model},
        )
        return results


def load_notes_from_file(filepath: str) -> list[ClinicalNote]:
    """
    Load clinical notes from a JSON file.

    Args:
        filepath: Path to the JSON file containing clinical notes.

    Returns:
        List of ClinicalNote objects.
    """
    with open(filepath, "r") as f:
        raw_notes = json.load(f)
    return [ClinicalNote(**note) for note in raw_notes]

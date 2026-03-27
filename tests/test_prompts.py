"""
Tests for prompt assembly and versioning (src/prompts.py).
Covers prompt retrieval, formatting, version routing, and content integrity.
"""

import pytest

from src.prompts import (
    build_user_message,
    get_few_shot_examples,
    get_system_prompt,
    SYSTEM_PROMPT_V1,
    FEW_SHOT_EXAMPLES_V1,
    USER_PROMPT_TEMPLATE_V1,
)


# =============================================================================
# get_system_prompt Tests
# =============================================================================

class TestGetSystemPrompt:
    """Tests for system prompt retrieval."""

    def test_returns_v1_prompt(self):
        prompt = get_system_prompt("v1.0")
        assert prompt == SYSTEM_PROMPT_V1

    def test_default_version_is_v1(self):
        prompt = get_system_prompt()
        assert prompt == SYSTEM_PROMPT_V1

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt version"):
            get_system_prompt("v99.0")

    def test_prompt_contains_urgency_scale(self):
        prompt = get_system_prompt("v1.0")
        assert "URGENCY SCALE" in prompt
        assert "1 = ROUTINE" in prompt
        assert "5 = EMERGENT" in prompt

    def test_prompt_contains_icd10_rules(self):
        prompt = get_system_prompt("v1.0")
        assert "ICD-10" in prompt

    def test_prompt_contains_safety_constraints(self):
        prompt = get_system_prompt("v1.0")
        assert "SAFETY CONSTRAINTS" in prompt
        assert "DECISION SUPPORT" in prompt

    def test_prompt_contains_json_output_format(self):
        prompt = get_system_prompt("v1.0")
        assert "urgency_level" in prompt
        assert "primary_complaint" in prompt
        assert "icd10_code" in prompt
        assert "reasoning" in prompt
        assert "recommended_action" in prompt


# =============================================================================
# get_few_shot_examples Tests
# =============================================================================

class TestGetFewShotExamples:
    """Tests for few-shot example retrieval."""

    def test_returns_v1_examples(self):
        examples = get_few_shot_examples("v1.0")
        assert examples == FEW_SHOT_EXAMPLES_V1

    def test_default_version_is_v1(self):
        examples = get_few_shot_examples()
        assert examples == FEW_SHOT_EXAMPLES_V1

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Unknown examples version"):
            get_few_shot_examples("v99.0")

    def test_examples_contain_three_urgency_levels(self):
        examples = get_few_shot_examples("v1.0")
        assert "Urgency 5" in examples
        assert "Urgency 3" in examples
        assert "Urgency 1" in examples

    def test_examples_contain_valid_json_structure(self):
        examples = get_few_shot_examples("v1.0")
        assert '"urgency_level"' in examples
        assert '"icd10_code"' in examples
        assert '"reasoning"' in examples


# =============================================================================
# build_user_message Tests
# =============================================================================

class TestBuildUserMessage:
    """Tests for user message formatting."""

    def test_note_text_included(self):
        note = "72-year-old male with chest pain."
        message = build_user_message(note, "v1.0")
        assert note in message

    def test_clinical_note_delimiters(self):
        note = "Test note content."
        message = build_user_message(note, "v1.0")
        assert "<clinical_note>" in message
        assert "</clinical_note>" in message

    def test_note_is_between_delimiters(self):
        note = "Patient has a cough."
        message = build_user_message(note, "v1.0")
        start = message.index("<clinical_note>")
        end = message.index("</clinical_note>")
        enclosed = message[start:end]
        assert note in enclosed

    def test_includes_json_instruction(self):
        message = build_user_message("Some note.", "v1.0")
        assert "JSON" in message

    def test_default_version_is_v1(self):
        message_default = build_user_message("Note text.")
        message_v1 = build_user_message("Note text.", "v1.0")
        assert message_default == message_v1

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Unknown template version"):
            build_user_message("Note text.", "v99.0")

    def test_special_characters_in_note(self):
        note = 'BP 120/80, temp 98.6F, HR 72. O2 sat 98%. {brackets} "quotes"'
        message = build_user_message(note, "v1.0")
        assert note in message

    def test_very_long_note(self):
        note = "A" * 10000
        message = build_user_message(note, "v1.0")
        assert note in message

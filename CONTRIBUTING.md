# Contributing to Clinical Note Classifier

Thank you for your interest in contributing. This project classifies free-text clinical notes into structured outputs (urgency, ICD-10 codes, reasoning) using LLMs with Pydantic validation. Contributions that improve classification accuracy, safety handling, or test coverage are especially welcome.

## Development Setup

```bash
git clone https://github.com/monfaredkavosh/clinical-note-classifier.git
cd clinical-note-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # or create .env manually
# Add your OpenAI API key: OPENAI_API_KEY=sk-...
```

## Running Tests

```bash
# Full test suite (123 tests)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Single module
python -m pytest tests/test_models.py -v
```

All tests must pass before submitting a pull request.

## Code Style

- **Type hints** on all function signatures and return types.
- **Pydantic v2 models** for all data structures (`src/models.py`). Do not use raw dicts for structured data.
- **Structured JSON logging** via `src/logging_config.py`. Use the configured logger, not `print()`.
- Keep prompts in `src/prompts.py` with version comments and dated entries.
- Follow existing patterns -- read the module you are modifying before making changes.

## Submitting Changes

1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature`).
2. Make your changes with tests covering new behavior.
3. Run the full test suite and confirm all tests pass.
4. Run the evaluation pipeline (`python -m src.main evaluate`) if your change affects classification logic or prompts.
5. Open a pull request against `main` with a clear description of what changed and why.

## Clinical Safety Considerations

This project handles clinical classification with patient safety implications. If your change modifies any of the following, take extra care:

- **Prompt text** (`src/prompts.py`) -- Any prompt change can shift urgency classifications. Run the full batch evaluation and compare metrics before and after.
- **Pydantic models** (`src/models.py`) -- Field constraints enforce clinical data integrity. Changes to validation rules (urgency range, ICD-10 format) require careful review.
- **Safety-first defaults** (`src/classifier.py`) -- The system defaults to urgency 5 (EMERGENT) on failure. Do not weaken this behavior.
- **Input validation** (`src/validation.py`) -- Guards against non-clinical content and prompt injection. Do not relax these checks without justification.

When in doubt, err on the side of patient safety. A false positive (over-triage) is always preferable to a false negative (missed critical case).

## Questions

Open an issue with the `[QUESTION]` prefix if you have questions about the codebase or contribution process.

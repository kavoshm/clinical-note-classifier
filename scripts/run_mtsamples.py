#!/usr/bin/env python3
"""
MTSamples Public Dataset Validation
=====================================
Loads the MTSamples clinical transcription dataset, selects a representative
sample of 50 notes across urgency-relevant specialties, converts them to the
classifier's expected format, runs classification, and produces a results CSV
with a summary table showing urgency distribution by specialty.

Usage:
    # With OPENAI_API_KEY set:
    python scripts/run_mtsamples.py

    # Custom CSV path:
    python scripts/run_mtsamples.py --csv /path/to/mtsamples.csv

    # Use pre-built sample (skip CSV loading):
    python scripts/run_mtsamples.py --use-sample

    # Custom model/provider:
    python scripts/run_mtsamples.py --model gpt-4o-mini --provider openai

Data source:
    MTSamples — 4,999 de-identified medical transcription samples (CC0 license)
    https://www.mtsamples.com/
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import ClinicalNoteClassifier, load_notes_from_file
from src.models import ClinicalNote, ClassifiedNote

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target specialties most relevant to clinical urgency classification.
# These cover emergency, acute, chronic, and routine care scenarios.
SPECIALTY_ALLOCATION = {
    "Emergency Room Reports": 8,
    "Cardiovascular / Pulmonary": 7,
    "General Medicine": 7,
    "Discharge Summary": 6,
    "Neurology": 5,
    "Gastroenterology": 4,
    "Hematology - Oncology": 4,
    "Nephrology": 3,
    "Psychiatry / Psychology": 3,
    "Pediatrics - Neonatal": 3,
}

DEFAULT_CSV_PATH = "/tmp/mtsamples.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "mtsamples_sample.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_CSV_PATH = OUTPUT_DIR / "mtsamples_results.csv"
RESULTS_JSON_PATH = OUTPUT_DIR / "mtsamples_classification.json"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data Loading and Sampling
# ---------------------------------------------------------------------------

def load_and_sample_mtsamples(csv_path: str, seed: int = RANDOM_SEED) -> list[dict]:
    """
    Load the MTSamples CSV, filter to urgency-relevant specialties,
    and select a stratified sample of 50 notes.

    Args:
        csv_path: Path to the mtsamples.csv file.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with keys: id, note, category.
    """
    random.seed(seed)

    notes_by_specialty: dict[str, list[dict]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spec = row["medical_specialty"].strip()
            text = (row.get("transcription") or "").strip()
            # Filter: must be a target specialty with valid transcription text
            if spec in SPECIALTY_ALLOCATION and text and len(text) >= 50:
                if spec not in notes_by_specialty:
                    notes_by_specialty[spec] = []
                # Use the CSV row index as the original ID
                row_index = row.get("", "0").strip()
                notes_by_specialty[spec].append({
                    "index": row_index,
                    "specialty": spec,
                    "transcription": text,
                })

    # Stratified sampling based on SPECIALTY_ALLOCATION
    selected = []
    for spec, count in SPECIALTY_ALLOCATION.items():
        pool = notes_by_specialty.get(spec, [])
        if not pool:
            print(f"  WARNING: No notes found for specialty '{spec}'")
            continue
        sampled = random.sample(pool, min(count, len(pool)))
        selected.extend(sampled)
        print(f"  {spec}: {len(sampled)} notes sampled (from {len(pool)} available)")

    # Convert to classifier input format
    classifier_notes = []
    for note in selected:
        classifier_notes.append({
            "id": f"mts_{note['index'].zfill(4)}",
            "note": note["transcription"],
            "category": note["specialty"],
        })

    return classifier_notes


def save_sample_json(notes: list[dict], output_path: Path) -> None:
    """Save the sampled notes as JSON in the classifier's expected format."""
    with open(output_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"\nSample saved to {output_path} ({len(notes)} notes)")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def run_classification(
    notes: list[dict],
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> list[ClassifiedNote]:
    """
    Run the classifier on a list of note dicts.

    Args:
        notes: List of dicts with keys: id, note, category.
        model: LLM model name.
        provider: LLM provider ("openai" or "anthropic").

    Returns:
        List of ClassifiedNote results.
    """
    clinical_notes = [ClinicalNote(**n) for n in notes]
    classifier = ClinicalNoteClassifier(model=model, provider=provider)

    def on_progress(current: int, total: int, note_id: str) -> None:
        print(f"  [{current}/{total}] Classifying {note_id}...")

    results = classifier.classify_batch(clinical_notes, on_progress=on_progress)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results_csv(results: list[ClassifiedNote], notes: list[dict], output_path: Path) -> None:
    """
    Save classification results as CSV with columns:
    id, specialty, note_preview, urgency, icd10_code, confidence_reasoning
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a lookup for specialty by note ID
    specialty_by_id = {n["id"]: n.get("category", "") for n in notes}

    fieldnames = [
        "id",
        "specialty",
        "note_preview",
        "urgency",
        "icd10_code",
        "confidence_reasoning",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            note_preview = r.note_text[:150].replace("\n", " ").replace("\r", " ")
            if len(r.note_text) > 150:
                note_preview += "..."
            writer.writerow({
                "id": r.note_id,
                "specialty": specialty_by_id.get(r.note_id, ""),
                "note_preview": note_preview,
                "urgency": f"{r.classification.urgency_level} ({r.classification.urgency_label})",
                "icd10_code": r.classification.icd10_code,
                "confidence_reasoning": r.classification.reasoning,
            })

    print(f"\nResults CSV saved to {output_path}")


def save_results_json(results: list[ClassifiedNote], output_path: Path) -> None:
    """Save full classification results as JSON (matching main.py batch format)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_data = []
    for r in results:
        entry = {
            "note_id": r.note_id,
            "classification": r.classification.model_dump(),
        }
        entry["classification"]["urgency_label"] = r.classification.urgency_label
        json_data.append(entry)

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Results JSON saved to {output_path}")


def print_summary_table(results: list[ClassifiedNote], notes: list[dict]) -> None:
    """
    Print a summary table showing urgency distribution by specialty.
    """
    specialty_by_id = {n["id"]: n.get("category", "") for n in notes}

    # Build urgency counts by specialty
    urgency_by_specialty: dict[str, Counter] = {}
    for r in results:
        spec = specialty_by_id.get(r.note_id, "Unknown")
        if spec not in urgency_by_specialty:
            urgency_by_specialty[spec] = Counter()
        urgency_by_specialty[spec][r.classification.urgency_level] += 1

    urgency_labels = {1: "ROUTINE", 2: "LOW", 3: "MODERATE", 4: "HIGH", 5: "EMERGENT"}

    print("\n" + "=" * 90)
    print("URGENCY DISTRIBUTION BY SPECIALTY")
    print("=" * 90)

    header = f"{'Specialty':<35} {'1-RTN':>6} {'2-LOW':>6} {'3-MOD':>6} {'4-HI':>6} {'5-EMR':>6} {'Total':>6}"
    print(header)
    print("-" * 90)

    # Overall totals
    overall = Counter()

    for spec in SPECIALTY_ALLOCATION:
        counts = urgency_by_specialty.get(spec, Counter())
        total = sum(counts.values())
        if total == 0:
            continue
        row = f"{spec:<35}"
        for level in [1, 2, 3, 4, 5]:
            c = counts.get(level, 0)
            overall[level] += c
            row += f" {c:>6}"
        row += f" {total:>6}"
        print(row)

    print("-" * 90)
    total_all = sum(overall.values())
    totals_row = f"{'TOTAL':<35}"
    for level in [1, 2, 3, 4, 5]:
        totals_row += f" {overall[level]:>6}"
    totals_row += f" {total_all:>6}"
    print(totals_row)
    print("=" * 90)

    # Overall distribution summary
    print("\nOVERALL URGENCY DISTRIBUTION:")
    for level in [1, 2, 3, 4, 5]:
        count = overall[level]
        pct = (count / total_all * 100) if total_all > 0 else 0
        bar = "#" * count
        print(f"  {level} ({urgency_labels[level]:>8s}): {bar} ({count}, {pct:.1f}%)")

    print(f"\nTotal notes classified: {total_all}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MTSamples public dataset validation against the clinical note classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_mtsamples.py
    python scripts/run_mtsamples.py --csv /path/to/mtsamples.csv
    python scripts/run_mtsamples.py --use-sample
    python scripts/run_mtsamples.py --model gpt-4o-mini --provider openai
        """,
    )

    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV_PATH,
        help=f"Path to mtsamples.csv (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--use-sample", action="store_true",
        help="Use pre-built sample from data/mtsamples_sample.json instead of loading CSV",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Only create the sample JSON, do not run classification",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MTSamples Public Dataset Validation")
    print("=" * 60)

    # Step 1: Load or sample notes
    if args.use_sample:
        print(f"\nLoading pre-built sample from {SAMPLE_PATH}...")
        with open(SAMPLE_PATH, "r") as f:
            notes = json.load(f)
        print(f"  Loaded {len(notes)} notes")
    else:
        print(f"\nLoading MTSamples CSV from {args.csv}...")
        if not Path(args.csv).exists():
            print(f"  ERROR: CSV file not found at {args.csv}")
            print("  Download MTSamples from https://www.mtsamples.com/ or provide --csv path")
            sys.exit(1)
        notes = load_and_sample_mtsamples(args.csv)
        # Also save the sample for future use
        save_sample_json(notes, SAMPLE_PATH)

    if args.sample_only:
        print("\n--sample-only flag set. Sample created. Exiting.")
        sys.exit(0)

    # Step 2: Check for API key
    api_key_var = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(api_key_var):
        print(f"\n  WARNING: {api_key_var} not found in environment.")
        print(f"  Set {api_key_var} and re-run to classify notes.")
        print(f"\n  Sample data is ready at: {SAMPLE_PATH}")
        print(f"  You can also run: python -m src.main batch --input {SAMPLE_PATH} --output outputs/")
        sys.exit(0)

    # Step 3: Run classification
    print(f"\nRunning classification with {args.provider}/{args.model}...")
    print(f"  Classifying {len(notes)} notes...\n")

    results = run_classification(notes, model=args.model, provider=args.provider)

    # Step 4: Save results
    print("\nSaving results...")
    save_results_csv(results, notes, RESULTS_CSV_PATH)
    save_results_json(results, RESULTS_JSON_PATH)

    # Step 5: Print summary
    print_summary_table(results, notes)

    print(f"\nDone. Results saved to:")
    print(f"  CSV:  {RESULTS_CSV_PATH}")
    print(f"  JSON: {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()

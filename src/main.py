"""
Clinical Note Classifier — CLI Entry Point
=============================================
Ties together the classifier, evaluation, and output formatting into a single
command-line interface. Supports:
  - Classifying a single note from text
  - Classifying all notes from the synthetic dataset
  - Running evaluation against expected outputs
  - Exporting results as JSON or CSV

Usage:
  python -m src.main classify --note "Patient presents with..."
  python -m src.main batch --input data/synthetic_notes.json --output outputs/
  python -m src.main evaluate --actual outputs/sample_classification.json --expected data/expected_outputs.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

from src.classifier import ClinicalNoteClassifier, load_notes_from_file
from src.evaluate import run_evaluation, evaluate_classifications, print_evaluation_report, load_json
from src.models import ClassifiedNote
from src.providers import SUPPORTED_PROVIDERS


def classify_single(
    note_text: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> None:
    """Classify a single clinical note and print the result."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    classifier = ClinicalNoteClassifier(model=model, provider=provider)

    if use_rich:
        console.print("[bold]Classifying clinical note...[/bold]\n")

    result = classifier.classify_note(note_text)

    output = result.model_dump()
    output["urgency_label"] = result.urgency_label
    formatted = json.dumps(output, indent=2)

    if use_rich:
        console.print(Panel(note_text[:300] + ("..." if len(note_text) > 300 else ""), title="Input Note"))
        syntax = Syntax(formatted, "json", theme="monokai")
        console.print(Panel(syntax, title="[bold green]Classification Result[/bold green]"))
    else:
        print(formatted)


def classify_batch(
    input_path: str,
    output_dir: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> None:
    """Classify all notes from an input file and save results."""
    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    notes = load_notes_from_file(input_path)
    classifier = ClinicalNoteClassifier(model=model, provider=provider)

    if use_rich:
        console.print(f"[bold]Classifying {len(notes)} clinical notes...[/bold]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Classifying", total=len(notes))

            def on_progress(current: int, total: int, note_id: str) -> None:
                progress.update(task, advance=1, description=f"Classifying {note_id}")

            results = classifier.classify_batch(notes, on_progress=on_progress)
    else:
        def on_progress(current: int, total: int, note_id: str) -> None:
            print(f"  [{current}/{total}] {note_id}")

        results = classifier.classify_batch(notes, on_progress=on_progress)

    # Save JSON output
    json_output_path = output_path / "sample_classification.json"
    json_data = []
    for r in results:
        entry = {
            "note_id": r.note_id,
            "classification": r.classification.model_dump(),
        }
        entry["classification"]["urgency_label"] = r.classification.urgency_label
        json_data.append(entry)

    with open(json_output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Save CSV output
    csv_output_path = output_path / "batch_results.csv"
    if results:
        fieldnames = list(results[0].to_summary_dict().keys())
        with open(csv_output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_summary_dict())

    if use_rich:
        console.print(f"\n[green]JSON results saved to:[/green] {json_output_path}")
        console.print(f"[green]CSV results saved to:[/green]  {csv_output_path}")
        console.print(f"\n[bold]Classified {len(results)} notes successfully.[/bold]")

        # Print urgency distribution
        from collections import Counter
        urgency_dist = Counter(r.classification.urgency_level for r in results)
        console.print("\n[bold]Urgency Distribution:[/bold]")
        for level in sorted(urgency_dist.keys()):
            bar = "#" * urgency_dist[level]
            label = {1: "ROUTINE", 2: "LOW", 3: "MODERATE", 4: "HIGH", 5: "EMERGENT"}
            console.print(f"  {level} ({label.get(level, '?'):>8s}): {bar} ({urgency_dist[level]})")
    else:
        print(f"Saved {len(results)} results to {output_dir}")


def run_evaluate(actual_path: str, expected_path: str) -> None:
    """Run evaluation comparing actual vs expected classifications."""
    run_evaluation(actual_path, expected_path)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Note Classifier — Classify and evaluate clinical notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main classify --note "72yo M with chest pain..."
  python -m src.main batch --input data/synthetic_notes.json --output outputs/
  python -m src.main evaluate --actual outputs/sample_classification.json --expected data/expected_outputs.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Classify single note
    classify_parser = subparsers.add_parser("classify", help="Classify a single clinical note")
    classify_parser.add_argument("--note", type=str, required=True, help="Clinical note text")
    classify_parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    classify_parser.add_argument(
        "--provider", type=str, default="openai",
        choices=list(SUPPORTED_PROVIDERS),
        help="LLM provider (default: openai)",
    )

    # Batch classify
    batch_parser = subparsers.add_parser("batch", help="Classify a batch of notes from file")
    batch_parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    batch_parser.add_argument("--output", type=str, required=True, help="Output directory path")
    batch_parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    batch_parser.add_argument(
        "--provider", type=str, default="openai",
        choices=list(SUPPORTED_PROVIDERS),
        help="LLM provider (default: openai)",
    )

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate classifications against expected")
    eval_parser.add_argument("--actual", type=str, required=True, help="Actual results JSON path")
    eval_parser.add_argument("--expected", type=str, required=True, help="Expected results JSON path")

    args = parser.parse_args()

    if args.command == "classify":
        classify_single(args.note, args.model, args.provider)
    elif args.command == "batch":
        classify_batch(args.input, args.output, args.model, args.provider)
    elif args.command == "evaluate":
        run_evaluate(args.actual, args.expected)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

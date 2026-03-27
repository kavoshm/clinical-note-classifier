"""
Evaluation Script for Clinical Note Classifier
================================================
Compares classifier outputs against expected (ground-truth) classifications.
Measures urgency level accuracy, ICD-10 code match rate, and generates a
detailed evaluation report.

This is essential for iterative prompt development: change the prompt, re-run
classification, re-run evaluation, compare metrics.
"""

import json
from pathlib import Path
from typing import Any, Optional

from src.models import EvaluationMetrics


def load_json(filepath: str) -> list[dict]:
    """Load a JSON file and return as list of dicts."""
    with open(filepath, "r") as f:
        return json.load(f)


def evaluate_classifications(
    actual_results: list[dict],
    expected_results: list[dict],
) -> EvaluationMetrics:
    """
    Compare actual classifier outputs against expected outputs.

    Args:
        actual_results: List of dicts with classifier output (must have 'id' or 'note_id' key).
        expected_results: List of dicts with expected output (must have 'id' key).

    Returns:
        EvaluationMetrics with accuracy measurements.
    """
    # Index expected results by ID for fast lookup
    expected_by_id: dict[str, dict] = {}
    for exp in expected_results:
        expected_by_id[exp["id"]] = exp

    total = 0
    urgency_exact = 0
    urgency_within_one = 0
    urgency_errors: list[float] = []
    icd10_matches = 0
    notes_evaluated: list[str] = []
    detailed_results: list[dict] = []

    for actual in actual_results:
        # Handle both 'id' and 'note_id' keys
        note_id = actual.get("note_id", actual.get("id", ""))
        if not note_id or note_id not in expected_by_id:
            continue

        expected = expected_by_id[note_id]
        total += 1
        notes_evaluated.append(note_id)

        # Extract actual values (handle nested 'classification' key)
        if "classification" in actual:
            actual_cls = actual["classification"]
        else:
            actual_cls = actual

        actual_urgency = actual_cls.get("urgency_level", -1)
        expected_urgency = expected.get("urgency_level", -1)
        actual_icd10 = actual_cls.get("icd10_code", "").strip()
        expected_icd10 = expected.get("icd10_code", "").strip()

        # Urgency comparison
        urgency_diff = abs(actual_urgency - expected_urgency)
        urgency_errors.append(urgency_diff)

        if urgency_diff == 0:
            urgency_exact += 1
            urgency_within_one += 1
        elif urgency_diff == 1:
            urgency_within_one += 1

        # ICD-10 comparison (exact match on code)
        if actual_icd10 == expected_icd10:
            icd10_matches += 1

        detailed_results.append({
            "note_id": note_id,
            "urgency_actual": actual_urgency,
            "urgency_expected": expected_urgency,
            "urgency_diff": urgency_diff,
            "icd10_actual": actual_icd10,
            "icd10_expected": expected_icd10,
            "icd10_match": actual_icd10 == expected_icd10,
        })

    mae = sum(urgency_errors) / len(urgency_errors) if urgency_errors else 0.0
    icd10_rate = icd10_matches / total if total > 0 else 0.0

    return EvaluationMetrics(
        total_notes=total,
        urgency_exact_match=urgency_exact,
        urgency_within_one=urgency_within_one,
        urgency_mae=mae,
        icd10_match_rate=icd10_rate,
        notes_evaluated=notes_evaluated,
    ), detailed_results


def print_evaluation_report(
    metrics: EvaluationMetrics,
    detailed: list[dict],
    use_rich: bool = True,
) -> None:
    """
    Print a formatted evaluation report.

    Args:
        metrics: Aggregated evaluation metrics.
        detailed: Per-note detailed comparison results.
        use_rich: Whether to use rich for pretty printing.
    """
    if use_rich:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich import box

            console = Console()
        except ImportError:
            use_rich = False

    if use_rich:
        console.print("\n[bold cyan]Clinical Note Classifier — Evaluation Report[/bold cyan]")
        console.print(f"[dim]Notes evaluated: {metrics.total_notes}[/dim]\n")

        # Summary metrics
        summary_table = Table(title="Summary Metrics", box=box.ROUNDED)
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", justify="right")

        summary_table.add_row(
            "Urgency Exact Match",
            f"{metrics.urgency_exact_match}/{metrics.total_notes} "
            f"({metrics.urgency_exact_accuracy:.1%})",
        )
        summary_table.add_row(
            "Urgency Within +/-1",
            f"{metrics.urgency_within_one}/{metrics.total_notes} "
            f"({metrics.urgency_within_one_accuracy:.1%})",
        )
        summary_table.add_row(
            "Urgency MAE",
            f"{metrics.urgency_mae:.2f}",
        )
        summary_table.add_row(
            "ICD-10 Exact Match",
            f"{metrics.icd10_match_rate:.1%}",
        )
        console.print(summary_table)

        # Detailed per-note results
        detail_table = Table(title="\nPer-Note Results", box=box.SIMPLE)
        detail_table.add_column("Note ID", style="cyan")
        detail_table.add_column("Urgency (Act.)", justify="center")
        detail_table.add_column("Urgency (Exp.)", justify="center")
        detail_table.add_column("Urg. Match", justify="center")
        detail_table.add_column("ICD-10 (Act.)")
        detail_table.add_column("ICD-10 (Exp.)")
        detail_table.add_column("ICD Match", justify="center")

        for d in detailed:
            urg_match = d["urgency_diff"] == 0
            detail_table.add_row(
                d["note_id"],
                str(d["urgency_actual"]),
                str(d["urgency_expected"]),
                "[green]exact[/green]" if urg_match else (
                    f"[yellow]+/-{d['urgency_diff']}[/yellow]" if d["urgency_diff"] == 1
                    else f"[red]+/-{d['urgency_diff']}[/red]"
                ),
                d["icd10_actual"],
                d["icd10_expected"],
                "[green]Y[/green]" if d["icd10_match"] else "[red]N[/red]",
            )

        console.print(detail_table)

        # Misclassifications summary
        misses = [d for d in detailed if d["urgency_diff"] > 1]
        if misses:
            console.print(
                f"\n[bold red]Significant urgency misclassifications (>1 level off): "
                f"{len(misses)} notes[/bold red]"
            )
            for m in misses:
                console.print(
                    f"  {m['note_id']}: actual={m['urgency_actual']} "
                    f"expected={m['urgency_expected']} (off by {m['urgency_diff']})"
                )
    else:
        print(f"\nEvaluation Report ({metrics.total_notes} notes)")
        print(f"Urgency Exact: {metrics.urgency_exact_accuracy:.1%}")
        print(f"Urgency +/-1:  {metrics.urgency_within_one_accuracy:.1%}")
        print(f"Urgency MAE:   {metrics.urgency_mae:.2f}")
        print(f"ICD-10 Match:  {metrics.icd10_match_rate:.1%}")


def run_evaluation(
    actual_path: str,
    expected_path: str,
) -> EvaluationMetrics:
    """
    Run evaluation from file paths.

    Args:
        actual_path: Path to actual classification results JSON.
        expected_path: Path to expected classification results JSON.

    Returns:
        EvaluationMetrics object.
    """
    actual = load_json(actual_path)
    expected = load_json(expected_path)

    metrics, detailed = evaluate_classifications(actual, expected)
    print_evaluation_report(metrics, detailed)

    return metrics


if __name__ == "__main__":
    import sys

    base_dir = Path(__file__).parent.parent

    actual_path = str(base_dir / "outputs" / "sample_classification.json")
    expected_path = str(base_dir / "data" / "expected_outputs.json")

    if len(sys.argv) >= 3:
        actual_path = sys.argv[1]
        expected_path = sys.argv[2]

    run_evaluation(actual_path, expected_path)

#!/usr/bin/env python3
"""
Generate Figures for Clinical Note Classifier
===============================================
Reads the actual project data files and generates polished matplotlib
visualizations for the README and documentation.

Usage:
    python scripts/generate_figures.py

Outputs saved to docs/images/:
    - urgency_distribution.png
    - classification_overview.png
    - accuracy_heatmap.png
    - pipeline_architecture.png
"""

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project colors — consistent palette across all figures
COLORS = {
    "blue": "#4f7cac",
    "teal": "#5a9e8f",
    "purple": "#9b6b9e",
    "orange": "#c47e3a",
    "red": "#b85450",
}

URGENCY_COLORS = {
    1: COLORS["teal"],
    2: COLORS["blue"],
    3: COLORS["orange"],
    4: COLORS["purple"],
    5: COLORS["red"],
}

URGENCY_LABELS = {
    1: "ROUTINE",
    2: "LOW",
    3: "MODERATE",
    4: "HIGH",
    5: "EMERGENT",
}

# Resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"


def load_data():
    """Load all data files needed for figure generation."""
    with open(DATA_DIR / "expected_outputs.json", "r") as f:
        expected = json.load(f)

    with open(OUTPUTS_DIR / "sample_classification.json", "r") as f:
        actual = json.load(f)

    with open(DATA_DIR / "synthetic_notes.json", "r") as f:
        notes = json.load(f)

    return expected, actual, notes


# ---------------------------------------------------------------------------
# Figure 1: Urgency Distribution
# ---------------------------------------------------------------------------

def generate_urgency_distribution(expected, actual):
    """
    Bar chart of urgency level distribution across the 20 notes,
    comparing expected vs actual classifications.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = [1, 2, 3, 4, 5]
    expected_counts = Counter(e["urgency_level"] for e in expected)
    actual_counts = Counter()
    for a in actual:
        cls = a.get("classification", a)
        actual_counts[cls["urgency_level"]] += 1

    expected_vals = [expected_counts.get(l, 0) for l in levels]
    actual_vals = [actual_counts.get(l, 0) for l in levels]

    x = np.arange(len(levels))
    bar_width = 0.35

    bars1 = ax.bar(
        x - bar_width / 2, expected_vals, bar_width,
        label="Expected", color=COLORS["blue"], edgecolor="white", linewidth=0.5,
        alpha=0.9,
    )
    bars2 = ax.bar(
        x + bar_width / 2, actual_vals, bar_width,
        label="Actual (GPT-4o-mini)", color=COLORS["teal"], edgecolor="white",
        linewidth=0.5, alpha=0.9,
    )

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 0.15,
                f"{int(height)}", ha="center", va="bottom",
                fontweight="bold", fontsize=11, color="white",
            )
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 0.15,
                f"{int(height)}", ha="center", va="bottom",
                fontweight="bold", fontsize=11, color="white",
            )

    ax.set_xlabel("Urgency Level", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Number of Notes", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        "Urgency Level Distribution Across 20 Clinical Notes",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{l}\n{URGENCY_LABELS[l]}" for l in levels],
        fontsize=11,
    )
    ax.set_ylim(0, max(max(expected_vals), max(actual_vals)) + 1.5)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "urgency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Created urgency_distribution.png")


# ---------------------------------------------------------------------------
# Figure 2: ICD-10 Category Distribution
# ---------------------------------------------------------------------------

def generate_classification_overview(expected):
    """
    Horizontal bar chart showing ICD-10 category distribution
    across the 20 clinical notes.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract ICD-10 top-level categories (first letter + description)
    icd10_category_map = {
        "I": "Circulatory System",
        "O": "Pregnancy/Childbirth",
        "J": "Respiratory System",
        "E": "Endocrine/Metabolic",
        "T": "Injury/Poisoning",
        "F": "Mental/Behavioral",
        "A": "Infectious Diseases",
        "R": "Symptoms/Signs",
        "N": "Genitourinary System",
        "Z": "Health Status/Services",
        "C": "Neoplasms",
        "G": "Nervous System",
        "M": "Musculoskeletal",
    }

    categories = Counter()
    for e in expected:
        code = e["icd10_code"]
        letter = code[0]
        cat_name = icd10_category_map.get(letter, f"Other ({letter})")
        categories[cat_name] += 1

    # Sort by count
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=False)
    cat_names = [c[0] for c in sorted_cats]
    cat_counts = [c[1] for c in sorted_cats]

    # Assign colors cycling through palette
    palette = list(COLORS.values())
    bar_colors = [palette[i % len(palette)] for i in range(len(cat_names))]

    bars = ax.barh(
        cat_names, cat_counts, color=bar_colors, edgecolor="white",
        linewidth=0.5, alpha=0.9, height=0.65,
    )

    # Add value labels
    for bar, count in zip(bars, cat_counts):
        ax.text(
            bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2.,
            f" {count}", ha="left", va="center",
            fontweight="bold", fontsize=11, color="white",
        )

    ax.set_xlabel("Number of Notes", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        "ICD-10 Category Distribution in Clinical Notes Dataset",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.set_xlim(0, max(cat_counts) + 1.2)
    ax.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "classification_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Created classification_overview.png")


# ---------------------------------------------------------------------------
# Figure 3: Accuracy Heatmap (Confusion Matrix)
# ---------------------------------------------------------------------------

def generate_accuracy_heatmap(expected, actual):
    """
    Heatmap comparing expected vs actual urgency levels
    (confusion matrix style).
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 7))

    # Build lookup for expected by note ID
    expected_by_id = {e["id"]: e["urgency_level"] for e in expected}

    # Build confusion matrix
    levels = [1, 2, 3, 4, 5]
    matrix = np.zeros((5, 5), dtype=int)

    for a in actual:
        note_id = a.get("note_id", a.get("id", ""))
        if note_id not in expected_by_id:
            continue
        cls = a.get("classification", a)
        actual_urgency = cls["urgency_level"]
        expected_urgency = expected_by_id[note_id]
        # matrix[row=expected, col=actual] (0-indexed)
        matrix[expected_urgency - 1][actual_urgency - 1] += 1

    # Create custom colormap from our palette
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "clinical", ["#1a1a2e", COLORS["blue"], COLORS["teal"], "#a8e6cf"], N=256
    )

    im = ax.imshow(matrix, cmap=cmap, aspect="equal", vmin=0)

    # Add text annotations
    for i in range(5):
        for j in range(5):
            val = matrix[i][j]
            color = "white" if val < matrix.max() * 0.7 else "#1a1a2e"
            fontweight = "bold" if i == j else "normal"
            ax.text(
                j, i, str(val), ha="center", va="center",
                fontsize=16, fontweight=fontweight, color=color,
            )

    # Draw diagonal highlight
    for i in range(5):
        rect = plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor=COLORS["teal"], linewidth=2, linestyle="--",
        )
        ax.add_patch(rect)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(
        [f"{l}\n{URGENCY_LABELS[l]}" for l in levels], fontsize=10,
    )
    ax.set_yticklabels(
        [f"{l} - {URGENCY_LABELS[l]}" for l in levels], fontsize=10,
    )
    ax.set_xlabel("Actual (Classifier Output)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Expected (Ground Truth)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        "Urgency Classification Confusion Matrix",
        fontsize=15, fontweight="bold", pad=15,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=11)

    # Compute and display accuracy
    correct = np.trace(matrix)
    total = np.sum(matrix)
    accuracy = correct / total * 100 if total > 0 else 0
    ax.text(
        0.02, -0.18,
        f"Exact Match Accuracy: {correct}/{total} ({accuracy:.0f}%)",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        color=COLORS["teal"],
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    fig.savefig(IMAGES_DIR / "accuracy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Created accuracy_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 4: Pipeline Architecture Diagram
# ---------------------------------------------------------------------------

def generate_pipeline_architecture():
    """
    Diagram of the classification pipeline using matplotlib
    (boxes and arrows): Clinical Note -> Prompt Assembly -> OpenAI API ->
    JSON Parse -> Structured Output
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Title
    ax.text(
        7, 4.6, "Clinical Note Classification Pipeline",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color="white",
    )

    # Pipeline stages
    stages = [
        {"x": 1.3, "label": "Clinical\nNote", "sublabel": "Free-text input", "color": COLORS["blue"]},
        {"x": 4.1, "label": "Prompt\nAssembly", "sublabel": "System + Few-shot\n+ CoT instructions", "color": COLORS["teal"]},
        {"x": 6.9, "label": "OpenAI\nAPI", "sublabel": "gpt-4o-mini\ntemp=0, JSON mode", "color": COLORS["purple"]},
        {"x": 9.7, "label": "JSON\nParse", "sublabel": "Pydantic\nvalidation", "color": COLORS["orange"]},
        {"x": 12.5, "label": "Structured\nOutput", "sublabel": "Urgency, ICD-10\nReasoning, Action", "color": COLORS["red"]},
    ]

    box_width = 2.0
    box_height = 2.2

    for stage in stages:
        # Main box
        fancy_box = FancyBboxPatch(
            (stage["x"] - box_width / 2, 1.6),
            box_width, box_height,
            boxstyle="round,pad=0.15",
            facecolor=stage["color"],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.85,
        )
        ax.add_patch(fancy_box)

        # Label text
        ax.text(
            stage["x"], 3.05, stage["label"],
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="white",
        )

        # Sublabel text
        ax.text(
            stage["x"], 2.15, stage["sublabel"],
            ha="center", va="center", fontsize=8.5,
            color="white", alpha=0.85, style="italic",
        )

    # Arrows between stages
    arrow_style = "Simple,tail_width=4,head_width=12,head_length=8"
    arrow_y = 2.7
    for i in range(len(stages) - 1):
        x_start = stages[i]["x"] + box_width / 2 + 0.05
        x_end = stages[i + 1]["x"] - box_width / 2 - 0.05
        arrow = FancyArrowPatch(
            (x_start, arrow_y), (x_end, arrow_y),
            arrowstyle=arrow_style,
            color="white",
            alpha=0.7,
            connectionstyle="arc3,rad=0",
        )
        ax.add_patch(arrow)

    # Retry loop indicator
    ax.annotate(
        "", xy=(5.1, 1.45), xytext=(8.7, 1.45),
        arrowprops=dict(
            arrowstyle="<-",
            color=COLORS["orange"],
            lw=1.5,
            linestyle="--",
            connectionstyle="arc3,rad=-0.4",
        ),
    )
    ax.text(
        6.9, 0.55, "Retry (up to 3x)\nwith exponential backoff",
        ha="center", va="center", fontsize=8, color=COLORS["orange"],
        style="italic", alpha=0.9,
    )

    # Source file references
    file_labels = [
        (1.3, 0.25, "input data"),
        (4.1, 0.25, "src/prompts.py"),
        (6.9, 0.25, "src/classifier.py"),
        (9.7, 0.25, "src/models.py"),
        (12.5, 0.25, "JSON + CSV"),
    ]
    for x, y, label in file_labels:
        ax.text(
            x, y, label, ha="center", va="center", fontsize=8,
            color="white", alpha=0.5, family="monospace",
        )

    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "pipeline_architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Created pipeline_architecture.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate all figures."""
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {IMAGES_DIR}\n")

    # Ensure output directory exists
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data files...")
    expected, actual, notes = load_data()
    print(f"  Loaded {len(expected)} expected outputs")
    print(f"  Loaded {len(actual)} actual classifications")
    print(f"  Loaded {len(notes)} clinical notes\n")

    # Generate all figures
    print("Generating figures...")
    generate_urgency_distribution(expected, actual)
    generate_classification_overview(expected)
    generate_accuracy_heatmap(expected, actual)
    generate_pipeline_architecture()

    print(f"\nAll figures saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()

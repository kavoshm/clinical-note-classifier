"""
Clinical Note Classifier — Live Demo (Gradio)
===============================================
Interactive web interface for classifying clinical notes into structured
outputs: urgency level, primary complaint, ICD-10 code, clinical reasoning,
and recommended actions.

Supports two modes:
  - Demo mode: Uses pre-computed classifications (no API key needed)
  - Live mode: Calls OpenAI API for real-time classification (requires key)

Deploy to HuggingFace Spaces or run locally:
  pip install gradio
  python app.py
"""

import json
import os
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Load pre-computed results for demo mode
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
OUTPUTS_DIR = Path(__file__).parent / "outputs"

# Load synthetic notes
with open(DATA_DIR / "synthetic_notes.json") as f:
    SYNTHETIC_NOTES = {note["id"]: note for note in json.load(f)}

# Load pre-computed classifications
with open(OUTPUTS_DIR / "sample_classification.json") as f:
    PRECOMPUTED = {item["note_id"]: item["classification"] for item in json.load(f)}

# Sample notes for the dropdown (short labels)
SAMPLE_LABELS = {
    "note_001": "72yo M — Crushing chest pain, ST elevation (Cardiac Emergency)",
    "note_002": "34yo F — 32wk pregnant, BP 168/108, HELLP (Obstetric Emergency)",
    "note_003": "8yo M — Sore throat, fever, strep+ (Pediatric Acute)",
    "note_004": "56yo F — Diabetes follow-up, A1C improving (Routine)",
    "note_005": "28yo M — Motorcycle crash, GCS 9 (Trauma Emergency)",
    "note_006": "42yo F — Depression, PHQ-9 score 19 (Mental Health)",
    "note_007": "78yo F — Altered mental status, UTI, sepsis (Geriatric Acute)",
    "note_008": "45yo M — Exertional dyspnea, BNP 1840 (Heart Failure)",
    "note_009": "16yo F — Fatigue, bruising, blasts on smear (Leukemia)",
    "note_010": "62yo M — Progressive dysphagia, weight loss (Oncology)",
    "note_011": "38yo M — Severe flank pain, hematuria (Renal Colic)",
    "note_012": "5mo F — Well-child visit, normal development (Routine)",
    "note_013": "55yo F — Thunderclap headache, neck stiffness (SAH)",
    "note_014": "70yo M — COPD exacerbation, SpO2 84% (Respiratory Emergency)",
    "note_015": "31yo F — Joint pain, RF+, anti-CCP+ (New-onset RA)",
    "note_016": "22yo M — Hallucinations, paranoia (First-Episode Psychosis)",
    "note_017": "48yo M — Diabetic foot ulcer probing to bone (Osteomyelitis)",
    "note_018": "85yo F — Progressive memory loss, MMSE 20/30 (Alzheimer's)",
    "note_019": "40yo F — Cough, night sweats, cavitary lesion (Tuberculosis)",
    "note_020": "52yo M — Palpitations, HR 168 irregular, AFib (Thyrotoxicosis)",
}

URGENCY_COLORS = {
    1: "#22c55e",  # green
    2: "#84cc16",  # lime
    3: "#eab308",  # yellow
    4: "#f97316",  # orange
    5: "#ef4444",  # red
}

URGENCY_EMOJI = {
    1: "1 — ROUTINE",
    2: "2 — LOW",
    3: "3 — MODERATE",
    4: "4 — HIGH",
    5: "5 — EMERGENT",
}


def format_result_html(classification: dict) -> str:
    """Format classification result as styled HTML."""
    urgency = classification["urgency_level"]
    color = URGENCY_COLORS.get(urgency, "#888")
    label = URGENCY_EMOJI.get(urgency, str(urgency))

    return f"""
    <div style="font-family: system-ui, -apple-system, sans-serif;">

      <div style="display:flex; align-items:center; gap:16px; margin-bottom:20px;">
        <div style="
          background:{color};
          color:#fff;
          font-weight:700;
          font-size:14px;
          padding:10px 20px;
          border-radius:10px;
          letter-spacing:0.05em;
          box-shadow: 0 2px 12px {color}44;
        ">URGENCY: {label}</div>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px;">
        <div style="background:rgba(0,0,0,0.03); border:1px solid rgba(0,0,0,0.08); border-radius:10px; padding:14px;">
          <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#888; margin-bottom:4px;">Primary Complaint</div>
          <div style="font-weight:600; color:#1a1a2e;">{classification['primary_complaint']}</div>
        </div>
        <div style="background:rgba(0,0,0,0.03); border:1px solid rgba(0,0,0,0.08); border-radius:10px; padding:14px;">
          <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#888; margin-bottom:4px;">ICD-10-CM Code</div>
          <div style="font-weight:600; color:#1a1a2e;">{classification['icd10_code']} — {classification.get('icd10_description', '')}</div>
        </div>
      </div>

      <div style="background:rgba(0,0,0,0.03); border:1px solid rgba(0,0,0,0.08); border-radius:10px; padding:14px; margin-bottom:12px;">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#888; margin-bottom:6px;">Clinical Reasoning</div>
        <div style="line-height:1.6; color:#333;">{classification['reasoning']}</div>
      </div>

      <div style="background:rgba(0,0,0,0.03); border:1px solid rgba(0,0,0,0.08); border-radius:10px; padding:14px;">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#888; margin-bottom:6px;">Recommended Action</div>
        <div style="line-height:1.6; color:#333;">{classification['recommended_action']}</div>
      </div>

    </div>
    """


def classify_note(note_text: str, api_key: str) -> tuple[str, str]:
    """Classify a clinical note. Uses live API if key provided, else demo mode."""
    if not note_text or not note_text.strip():
        return "<p style='color:#888;'>Please enter a clinical note or select a sample.</p>", ""

    # Check if this is a known synthetic note (for demo mode)
    demo_note_id = None
    for nid, note_data in SYNTHETIC_NOTES.items():
        if note_data["note"].strip() == note_text.strip():
            demo_note_id = nid
            break

    # Try live classification if API key provided
    if api_key and api_key.strip().startswith("sk-"):
        try:
            os.environ["OPENAI_API_KEY"] = api_key.strip()
            from src.classifier import ClinicalNoteClassifier
            classifier = ClinicalNoteClassifier(model="gpt-4o-mini")
            result = classifier.classify_note(note_text)
            classification = result.model_dump()
            classification["urgency_label"] = result.urgency_label
            mode = "Live classification via OpenAI gpt-4o-mini"
            return format_result_html(classification), mode
        except Exception as e:
            if demo_note_id and demo_note_id in PRECOMPUTED:
                mode = f"Demo mode (API error: {str(e)[:80]}... falling back to pre-computed)"
                return format_result_html(PRECOMPUTED[demo_note_id]), mode
            return f"<p style='color:#ef4444;'>Classification failed: {str(e)}</p>", "Error"

    # Demo mode — use pre-computed results
    if demo_note_id and demo_note_id in PRECOMPUTED:
        return format_result_html(PRECOMPUTED[demo_note_id]), "Demo mode (pre-computed results)"

    return (
        "<p style='color:#888;'>No API key provided and this note doesn't match a sample. "
        "Enter an OpenAI API key for live classification, or select a sample note.</p>",
        "",
    )


def load_sample(sample_label: str) -> str:
    """Load a sample note by its label."""
    if not sample_label:
        return ""
    for nid, label in SAMPLE_LABELS.items():
        if label == sample_label:
            return SYNTHETIC_NOTES[nid]["note"]
    return ""


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Clinical Note Classifier") as demo:
    gr.HTML("""
        <div class="main-header">
            <h1>Clinical Note Classifier</h1>
            <p style="color:#6b7280; font-size:15px;">
                LLM-powered clinical decision support — urgency triage, ICD-10 coding, and chain-of-thought reasoning
            </p>
        </div>
        <div class="disclaimer">
            <strong>Decision Support Tool Only</strong> — Not for clinical use.
            All outputs are suggestive, not definitive. Uses synthetic data for demonstration.
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLE_LABELS.values()),
                label="Sample Clinical Notes (20 cases)",
                info="Select a pre-loaded case or type your own below",
            )
            note_input = gr.Textbox(
                label="Clinical Note",
                placeholder="Paste a clinical note here, or select a sample above...",
                lines=10,
                max_lines=20,
            )
            api_key_input = gr.Textbox(
                label="OpenAI API Key (optional)",
                placeholder="sk-... (leave blank for demo mode with pre-computed results)",
                type="password",
            )
            classify_btn = gr.Button("Classify Note", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_html = gr.HTML(
                value="<p style='color:#888; text-align:center; padding:40px;'>Select a sample note or enter your own to begin.</p>",
                label="Classification Result",
            )
            mode_text = gr.Textbox(label="Mode", interactive=False, elem_classes=["mode-badge"])

    # Event handlers
    sample_dropdown.change(fn=load_sample, inputs=[sample_dropdown], outputs=[note_input])
    classify_btn.click(
        fn=classify_note,
        inputs=[note_input, api_key_input],
        outputs=[result_html, mode_text],
    )

    gr.HTML("""
        <div style="text-align:center; margin-top:24px; padding:16px; border-top:1px solid #e5e7eb;">
            <p style="font-size:13px; color:#9ca3af;">
                Built by <a href="https://github.com/kavoshm" style="color:#3b82f6;">Kavosh Monfared</a> —
                <a href="https://github.com/kavoshm/clinical-note-classifier" style="color:#3b82f6;">View Source</a> —
                100% urgency accuracy on 20 test cases · 85% ICD-10 exact match · 123 unit tests
            </p>
            <p style="font-size:12px; color:#b0b0b0; margin-top:4px;">
                gpt-4o-mini · Temperature 0 · JSON Mode + Pydantic validation · Safety-first defaults (urgency 5 on error)
            </p>
        </div>
    """)


if __name__ == "__main__":
    demo.launch()

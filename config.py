"""config.py: Global configuration, prompts, and output utilities for the pipeline."""

import os
import csv
import json
from typing import Any, Dict

DATA_PATH = "data/Financial-QA-10k.csv"
OUTPUT_DIR = "outputs"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# LLMs
MODEL_A_NAME = "google/gemini-3.1-flash-lite-preview"
MODEL_A_LABEL = "gemini-3.1-flash-lite"
MODEL_B_NAME = "openai/gpt-5.4-mini"
MODEL_B_LABEL = "gpt-5.4-mini"
MODEL_C_NAME = "mistralai/mistral-large-2512"
MODEL_C_LABEL = "mistral-large-2512"
JUDGE_MODEL = "google/gemini-3-flash-preview"

# Parameters
N_SAMPLES = 50
MAX_TOKENS = 512
TEMPERATURE = 0.0
THRESHOLD = 0.50

SYSTEM_PROMPT = """
You are a highly precise financial extraction assistant.
Your task is to answer questions strictly based on the context provided.

RULES:
1. Use ONLY information explicitly stated in the context. Do NOT use external knowledge.
2. If the context does not contain sufficient information to answer, set "abstain" to true.
3. Keep your answer concise (1-3 sentences maximum). Extract phrasing strictly from the text when possible.
4. Return ONLY a valid JSON object.

RESPONSE FORMAT:
{
  "answer": "<concise answer, or empty string if abstaining>",
  "confidence": <float 0.0 to 1.0 reflecting support quality>,
  "abstain": <true if context is insufficient, false otherwise>,
  "reason": "<reason for abstaining, or empty>"
}
"""


def compare_metrics(
    summary_a: Dict[str, Any], summary_b: Dict[str, Any], summary_c: Dict[str, Any]
):
    """Print a side-by-side comparison of model metrics."""
    print(
        f"\n{'Metric':<20} | {summary_a['model_label']:<20} | {summary_b['model_label']:<20} | {summary_c['model_label']:<20}"
    )
    for key in ["llm_as_a_judge", "cosine_sim", "mean_confidence", "abstention_rate"]:
        print(
            f"{key:<20} | {summary_a.get(key, 0):<20.4f} | {summary_b.get(key, 0):<20.4f} | {summary_c.get(key, 0):<20.4f}"
        )


def save_outputs(preds_a, preds_b, preds_c, sum_a, sum_b, sum_c):
    """Save predictions and evaluation summaries to the outputs directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for p_list, label in [
        (preds_a, MODEL_A_LABEL),
        (preds_b, MODEL_B_LABEL),
        (preds_c, MODEL_C_LABEL),
    ]:
        with open(
            os.path.join(OUTPUT_DIR, f"predictions_{label}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                [{k: v for k, v in p.items() if k != "context"} for p in p_list],
                f,
                indent=2,
            )

    columns = [
        "id",
        "ticker",
        "model_label",
        "question",
        "reference_answer",
        "answer",
        "abstain",
        "confidence",
        "llm_as_a_judge",
        "cosine_sim",
        "reason",
    ]
    with open(
        os.path.join(OUTPUT_DIR, "evaluation_summary.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(
                preds_a + preds_b + preds_c, key=lambda x: (x["id"], x["model_label"])
            )
        )

    with open(
        os.path.join(OUTPUT_DIR, "model_comparison.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", MODEL_A_LABEL, MODEL_B_LABEL, MODEL_C_LABEL]
        )
        writer.writeheader()
        for key in [
            "n_samples",
            "llm_as_a_judge",
            "cosine_sim",
            "mean_confidence",
            "abstention_rate",
        ]:
            writer.writerow(
                {
                    "metric": key,
                    MODEL_A_LABEL: sum_a.get(key, ""),
                    MODEL_B_LABEL: sum_b.get(key, ""),
                    MODEL_C_LABEL: sum_c.get(key, ""),
                }
            )

    print(f"\nOutputs saved to {OUTPUT_DIR}/")

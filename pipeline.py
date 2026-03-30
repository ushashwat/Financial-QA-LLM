"""pipeline.py: Core application logic for generative loading, inference, and evaluation."""

import os
import csv
import json
import asyncio
from typing import Any, Dict, List, Tuple
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

import config as cfg


def load_samples() -> List[Dict[str, Any]]:
    """Load a fixed number of samples from the CSV dataset."""
    samples = []
    with open(cfg.DATA_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= cfg.N_SAMPLES:
                break
            samples.append(
                {
                    "id": i + 1,
                    "question": row.get("question", "").strip(),
                    "reference_answer": row.get("answer", "").strip(),
                    "context": row.get("context", "").strip(),
                    "ticker": row.get("ticker", "").strip(),
                    "filing": row.get("filing", "").strip(),
                }
            )
    return samples


def _parse_response(raw_text: str) -> Dict[str, Any]:
    """Parse and validate JSON response from the LLM."""
    try:
        data = json.loads(raw_text)
        ans = str(data.get("answer", "")).strip()
        conf = float(data.get("confidence", 0.0))
        abst = bool(data.get("abstain", False)) or (conf < cfg.THRESHOLD)
        return {
            "answer": ans,
            "confidence": conf,
            "abstain": abst,
            "reason": (
                "Threshold"
                if abst and conf < cfg.THRESHOLD
                else str(data.get("reason", "")).strip()
            ),
        }
    except json.JSONDecodeError:
        return {"answer": "", "confidence": 0.0, "abstain": True, "reason": "Bad JSON"}


async def _fetch(
    client: AsyncOpenAI,
    model_name: str,
    model_label: str,
    sample: Dict[str, Any],
    i: int,
    total: int,
) -> Dict[str, Any]:
    """Execute a single async inference call with output parsing."""
    user_msg = f"CONTEXT:\n{sample['context']}\n\nQUESTION:\n{sample['question']}\n\nRespond with strict JSON."
    try:
        res = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": cfg.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=cfg.MAX_TOKENS,
            temperature=cfg.TEMPERATURE,
            response_format={"type": "json_object"},
        )
        result = _parse_response(res.choices[0].message.content or "")
        status = "OK" if not result["abstain"] else "ABSTAIN"
        print(f"  [{i}/{total}] {sample['ticker']} | {status}", flush=True)
        return {**sample, "model": model_name, "model_label": model_label, **result}
    except Exception:
        print(f"  [{i}/{total}] {sample['ticker']} | FAILED", flush=True)
        return {
            **sample,
            "model": model_name,
            "model_label": model_label,
            "answer": "",
            "confidence": 0.0,
            "abstain": True,
            "reason": "API Failure",
        }


async def run_inference(
    model_name: str, model_label: str, samples: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Execute API calls concurrently via asyncio."""
    client = AsyncOpenAI(
        base_url=cfg.OPENROUTER_BASE_URL, api_key=os.environ["API_KEY"]
    )
    print(f"\n--- Running Async Inference: {model_label} ({model_name}) ---")
    tasks = [
        _fetch(client, model_name, model_label, s, i + 1, len(samples))
        for i, s in enumerate(samples)
    ]
    return await asyncio.gather(*tasks)


def _cosine_sim(pred: str, ref: str) -> float:
    """Calculate cosine similarity using bag-of-words mapping."""
    if not pred.strip() or not ref.strip():
        return 0.0
    vocab = list(set(pred.lower().split()) | set(ref.lower().split()))
    v1 = [[pred.lower().split().count(w) for w in vocab]]
    v2 = [[ref.lower().split().count(w) for w in vocab]]
    return round(float(cos_sim(v1, v2)[0][0]), 4)


async def _judge(
    client: AsyncOpenAI, pred: str, ref: str, i: int = 0, total: int = 0
) -> int:
    """Evaluate factual equivalence using LLM-as-a-judge method."""
    if not pred.strip() or not ref.strip():
        return 0
    msg = f'REFERENCE:\n{ref}\n\nPREDICTION:\n{pred}\n\nStrictly reply with valid JSON format: {{"factual_match": 1}} if facts match, else {{"factual_match": 0}}.'
    try:
        res = await asyncio.wait_for(
            client.chat.completions.create(
                model=cfg.JUDGE_MODEL,
                messages=[{"role": "user", "content": msg}],
                max_tokens=32,
                temperature=0.0,
                response_format={"type": "json_object"},
            ),
            timeout=15.0,
        )
        score = int(
            json.loads(res.choices[0].message.content or "{}").get("factual_match", 0)
        )
        if i and total:
            print(
                f"  [judge {i}/{total}] {'MATCH' if score else 'NO MATCH'}",
                flush=True,
            )
        return score
    except Exception:
        if i and total:
            print(f"  [judge {i}/{total}] FAILED (returning 0)", flush=True)
        return 0


async def evaluate(
    predictions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate factual correctness via LLM-as-a-Judge and cosine similarity."""
    client = AsyncOpenAI(
        base_url=cfg.OPENROUTER_BASE_URL, api_key=os.environ["API_KEY"]
    )
    n = len(predictions)
    print(f"Running LLM-as-a-Judge for {n} predictions...", flush=True)

    judge_tasks = [
        _judge(
            client,
            p["answer"] if not p["abstain"] else "",
            p["reference_answer"],
            i + 1,
            n,
        )
        for i, p in enumerate(predictions)
    ]
    judge_scores = await asyncio.gather(*judge_tasks)

    for i, p in enumerate(predictions):
        p["llm_as_a_judge"] = judge_scores[i]
        p["cosine_sim"] = _cosine_sim(
            p["answer"] if not p["abstain"] else "", p["reference_answer"]
        )

    answered = [p for p in predictions if not p["abstain"]]
    summary = {
        "model_label": predictions[0]["model_label"] if predictions else "",
        "n_samples": n,
        "n_answered": len(answered),
        "n_abstained": n - len(answered),
        "llm_as_a_judge": (
            round(sum(p["llm_as_a_judge"] for p in predictions) / n, 4) if n else 0.0
        ),
        "cosine_sim": (
            round(sum(p["cosine_sim"] for p in predictions) / n, 4) if n else 0.0
        ),
        "mean_confidence": (
            round(sum(p["confidence"] for p in predictions) / n, 4) if n else 0.0
        ),
        "abstention_rate": round(1 - len(answered) / n, 4) if n else 0.0,
    }
    return predictions, summary

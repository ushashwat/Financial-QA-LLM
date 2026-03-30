"""main.py: Entry point for the Financial QA evaluation pipeline."""

import os
from dotenv import load_dotenv
import asyncio
import pipeline
import config as cfg

load_dotenv()


async def main():
    assert os.environ.get("API_KEY"), "API_KEY environment variable not set!"

    print("Financial QA Pipeline")

    samples = pipeline.load_samples()
    print(f"Loaded {len(samples)} samples.")

    preds_a = await pipeline.run_inference(cfg.MODEL_A_NAME, cfg.MODEL_A_LABEL, samples)
    preds_b = await pipeline.run_inference(cfg.MODEL_B_NAME, cfg.MODEL_B_LABEL, samples)
    preds_c = await pipeline.run_inference(cfg.MODEL_C_NAME, cfg.MODEL_C_LABEL, samples)

    print("\nEvaluating Predictions: ")
    results_a, sum_a = await pipeline.evaluate(preds_a)
    results_b, sum_b = await pipeline.evaluate(preds_b)
    results_c, sum_c = await pipeline.evaluate(preds_c)

    cfg.compare_metrics(sum_a, sum_b, sum_c)
    cfg.save_outputs(results_a, results_b, results_c, sum_a, sum_b, sum_c)


if __name__ == "__main__":
    asyncio.run(main())

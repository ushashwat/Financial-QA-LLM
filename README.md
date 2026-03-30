# Financial QA Evaluation Pipeline
A lightweight LLM-based question-answering pipeline built for evaluating candidate models on financial 10-K data.

## Overview
This project evaluates three LLMs on a 50-sample subset of the Financial QA 10-K dataset, comparing their ability to answer questions **grounded strictly in the provided context** from SEC 10-K filings.

### LLM comparison
| Label | Model ID | Provider |
|---|---|---|
| `gemini-3.1-flash-lite` | `google/gemini-3.1-flash-lite-preview` | Google via OpenRouter |
| `gpt-5.4-mini` | `openai/gpt-5.4-mini` | OpenAI via OpenRouter |
| `mistral-large-2512` | `mistralai/mistral-large-2512` | Mistral via OpenRouter |

**Judge model** (for LLM-as-a-Judge metric only): `google/gemini-3-flash-preview`

## Pipeline Architecture
```
main.py (entry point)
│
data/Financial-QA-10k.csv  (50 rows)
│
pipeline.py (loads & normalises samples)
│
pipeline.py (calls OpenRouter API concurrently, parses JSON)
│
Model A, Model B, Model C
│
pipeline.py (LLM-as-a-judge, Cosine Similarity)
│
config.py (writes JSON + CSV outputs)
│
outputs/ (contains all saved outputs)
```

## Design Decisions
**Sample Selection:** First 50 rows of the dataset. Random sampling would be more statistically rigorous but adds reproducibility complexity (seed management) and would produce overhead for this task.

**Model Selection:** These models provide an excellent footprint across the lightweight and highly capable classes of LLMs. Comparing them provides a realistic look at cost vs. performance decisions in production. OpenRouter provides a single, OpenAI-compatible endpoint, simplifying the client code.

**Prompt Design:** Single system prompt enforcing JSON output, context-grounding, and abstention as the criteria:
- **JSON output** makes parsing reliable and the schema explicit.
- **Context-grounding instruction** guards against hallucination.
- **Abstention as a category** is preferable to a low-confidence answer - in financial QA, a wrong answer can be more damaging than no answer.
- **Temperature = 0** ensures deterministic, reproducible outputs.

**Confidence & Abstention:** Self-reported confidence (0–1) with a threshold of 0.50. If a model answers but with very low confidence, it is reclassified as abstain regardless of its stated answer.

## Assumptions
- Reference answers in the dataset are treated as ground truth without independent validation.
- Context provided in each row is assumed to be the correct and complete context for that question.
- Retry logic is not implemented for API calls, so OpenRouter or another LLM platform uptime is expected.
- No tool calling, RAG, or MCP logic was used in any of the implemented models.

## Evaluation Metrics
| Metric | Reasoning | Limitation |
|---|---|---|
| LLM as a Judge | Evaluates factual equivalence vs reference using a stronger LLM | Extra API call overhead per sample |
| Cosine Similarity | Bag-of-Words overlap; no external dependencies | Penalises paraphrasing and vocabulary shifts |
| Mean Confidence | Trustworthiness signal of self-reported scores | Self-reported values are not well calibrated |
| Abstention Rate | Coverage vs safety trade-off | Does not capture quality of abstentions |

**LLM-as-a-Judge is the primary recommendation metric** as it measures factual equivalence directly, bypassing the vocabulary limitations of Cosine Similarity.

### Results
| Metric | gemini-3.1-flash-lite | gpt-5.4-mini | mistral-large-2512 |
|---|---|---|---|
| LLM as a Judge | 0.94 | 1.00 | 0.94 |
| Cosine Similarity | 0.77 | 0.68 | 0.74 |
| Mean Confidence | 1.00 | 0.98 | 0.98 |
| Abstention Rate | 0.02 | 0.00 | 0.02 |

### Model Performance
- Gemini had the **best average performance overall** and gave the most semantically similar answers.
- GPT was found to be slightly less verbose in its response, hence the lower Cosine Similarity score.
- Mistral LLM has the lowest parameter count out of the three but still performs comparatively well.
- All three LLMs gave **deterministic results** due to structured output schema and strict prompting.

## Quick Start
### Prerequisites
- Python 3.13
- An **OpenRouter API key** (stored as `API_KEY` in your `.env` file)

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Run the pipeline
```bash
python main.py
```

This will:
1. Load the first 50 samples from `data/Financial-QA-10k.csv`
2. Run concurrent inference with Models A, B, and C via OpenRouter
3. Evaluate all models via LLM-as-a-Judge and Cosine Similarity
4. Print a comparison table to the console
5. Save all outputs to `outputs/`

### Interactive Pipeline (Optional)
Open the Jupyter notebook (`notebook.ipynb`) and run all cells in order. It produces the same outputs as `main.py`, but with visualisation charts saved to `outputs/`.
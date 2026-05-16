# Unified RAGAS Evaluation: Variation Comparison

A research notebook that implements builds on top of the [RAGAS](https://arxiv.org/abs/2309.15217) (Retrieval-Augmented Generation Assessment) evaluation framework, side-by-side, to measure RAG pipeline quality.

---

## Overview

| | V1 — Improved (SBERT) | V2 — Standard (Gemini) |
|---|---|---|
| **Embeddings** | `all-mpnet-base-v2` with multi-layer pooling | `gemini-embedding-001` via API |
| **Metrics** | Faithfulness, Answer Relevancy, Context Relevancy, Answer Correctness | Faithfulness, Answer Relevancy, Context Relevancy |
| **LLM Backend** | Groq (`llama-3.3-70b-versatile`) | Groq (`llama-3.3-70b-versatile`) |
| **Key Extra** | Ablation study + PCA centering | Baseline reference implementation |

---

## Metrics Implemented

- **Faithfulness** — checks whether the generated answer is grounded in the retrieved context, using LLM-based claim verification.
- **Answer Relevancy** — measures how well the answer addresses the original question via reverse question generation and embedding similarity.
- **Context Relevancy** — quantifies what fraction of the retrieved context is actually relevant to the question.
- **Answer Correctness** *(V1 only)* — fact-overlap comparison between the generated answer and a ground-truth reference.

---

## Architecture Highlights (V1)

V1 goes beyond the reference RAGAS paper with two improvements:

**Multi-layer hidden state pooling** — instead of using only the final transformer layer, embeddings are computed by averaging the last 4 hidden layers of `all-mpnet-base-v2`, which produces richer semantic representations.

**PCA-style mean centering** — embedding vectors are centered against a corpus-wide global mean before similarity is computed, reducing bias and improving sensitivity between high- and low-quality answers.

An ablation study (`run_ablation_study`) tracks scores across 6 progressive configurations (Base → Pool → +Sharpening → +PCA) to quantify the contribution of each improvement.

---

## Setup

### Prerequisites

```bash
pip install groq sentence-transformers scikit-learn google-generativeai \
            transformers torch nltk plotly seaborn pandas
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### Test Data

The notebook expects a `paper_tests.json` file in the working directory with the following structure:

```json
[
  {
    "metric": "Faithfulness",
    "question": "...",
    "context": "...",
    "high_answer": "...",
    "low_answer": "..."
  },
  {
    "metric": "Answer Relevancy",
    "question": "...",
    "high_answer": "...",
    "low_answer": "..."
  },
  {
    "metric": "Context Relevancy",
    "question": "...",
    "high_context": "...",
    "low_context": "..."
  }
]
```

---

## Usage

Open and run `unified_ragas_comparison.ipynb` top-to-bottom. The notebook will:

1. Initialize both evaluators (V1 and V2)
2. Run side-by-side metric comparisons on your test cases
3. Execute the ablation study on Answer Relevancy
4. Output two summary tables and three visualizations

---

## Outputs

**Table 1 — Architectural Ablation Study**: scores for each of the 6 embedding configurations on high vs. low relevancy samples.

**Table 2 — Unified Performance Results**: V1 vs. V2 scores across all metrics and cases.

**Visualizations** (via Plotly + Seaborn):
- Grouped bar chart: V1 vs. V2 performance comparison
- Heatmap: ablation sensitivity across configurations
- Line chart: score delta ("gap") between high and low quality answers per architecture step

---

## References

- Es, S., James, J., Anke, L. E., & Schockaert, S. (2024). [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217). EACL 2024.

# RAG Evaluation Results

## Framework

- Selected framework: DeepEval
- Runtime status: used
- Fallback scorer: deterministic token-overlap metrics with the same four rubric names.

## Overall Scores

| Metric | Config A: hybrid + rerank | Config B: hybrid without rerank | Delta |
|---|---:|---:|---:|
| faithfulness | 0.911 | 0.915 | -0.004 |
| answer_relevance | 0.837 | 0.823 | +0.014 |
| context_recall | 0.594 | 0.608 | -0.014 |
| context_precision | 0.494 | 0.550 | -0.056 |
| average | 0.709 | 0.724 | -0.015 |
| latency_seconds | 2.060 | 0.465 | +1.595 |

## A/B Comparison Analysis

- Config A uses semantic search, BM25 lexical search, RRF merge, and reranking.
- Config B uses the same hybrid retrieval stack without reranking.
- Conclusion: Config B has the better average score in this run.

## Worst Performers (Bottom 3)

| # | Config | Category | Question | Average | Likely root cause |
|---:|---|---|---|---:|---|
| 1 | B_hybrid_no_rerank | safety | Nếu câu hỏi yêu cầu thông tin không có trong corpus thì hệ thống nên trả lời thế nào? | 0.483 | Retriever missed exact expected context or answer required information outside top chunks. |
| 2 | A_hybrid_rerank | safety | Nếu câu hỏi yêu cầu thông tin không có trong corpus thì hệ thống nên trả lời thế nào? | 0.496 | Retriever missed exact expected context or answer required information outside top chunks. |
| 3 | A_hybrid_rerank | mixed | Pipeline cần trích dẫn loại nguồn nào khi trả lời câu hỏi về pháp luật ma túy và tin nghệ sĩ? | 0.543 | Retriever missed exact expected context or answer required information outside top chunks. |

## Recommendations

1. Add more criminal-code documents so questions about penalties and prosecution stages have stronger legal grounding.
2. Keep reranking enabled for demo because it generally improves source ordering and context precision.
3. Expand article metadata with publication dates and named entities to improve mixed legal-news questions.

## Evaluation Details

- Golden dataset size: 16
- Total evaluated rows: 32
- Raw per-question details: `group_project/evaluation/eval_details.json`

# RAG Evaluation Results

## Framework

- Selected framework: DeepEval
- Runtime status: used
- Fallback scorer: deterministic token-overlap metrics with the same four rubric names.

## Overall Scores

| Metric | Config A: hybrid + rerank | Config B: hybrid without rerank | Delta |
|---|---:|---:|---:|
| faithfulness | 0.846 | 0.849 | -0.003 |
| answer_relevance | 0.753 | 0.735 | +0.018 |
| context_recall | 0.582 | 0.576 | +0.006 |
| context_precision | 0.470 | 0.461 | +0.009 |
| average | 0.663 | 0.655 | +0.008 |
| latency_seconds | 5.516 | 12.021 | -6.505 |

## A/B Comparison Analysis

- Config A uses semantic search, BM25 lexical search, RRF merge, and reranking.
- Config B uses the same hybrid retrieval stack without reranking.
- Conclusion: Config A has the better average score in this run.

## Worst Performers (Bottom 3)

| # | Config | Category | Question | Average | Likely root cause |
|---:|---|---|---|---:|---|
| 1 | A_hybrid_rerank | safety | Nếu câu hỏi yêu cầu thông tin không có trong corpus thì hệ thống nên trả lời thế nào? | 0.412 | Retriever missed exact expected context or answer required information outside top chunks. |
| 2 | B_hybrid_no_rerank | safety | Nếu câu hỏi yêu cầu thông tin không có trong corpus thì hệ thống nên trả lời thế nào? | 0.412 | Retriever missed exact expected context or answer required information outside top chunks. |
| 3 | A_hybrid_rerank | mixed | Pipeline cần trích dẫn loại nguồn nào khi trả lời câu hỏi về pháp luật ma túy và tin nghệ sĩ? | 0.420 | Retriever missed exact expected context or answer required information outside top chunks. |

## Recommendations

1. Add more criminal-code documents so questions about penalties and prosecution stages have stronger legal grounding.
2. Keep reranking enabled for demo because it generally improves source ordering and context precision.
3. Expand article metadata with publication dates and named entities to improve mixed legal-news questions.

## Evaluation Details

- Golden dataset size: 16
- Total evaluated rows: 32
- Raw per-question details: `group_project/evaluation/eval_details.json`

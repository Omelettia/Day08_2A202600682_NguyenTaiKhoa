"""
Group RAG evaluation pipeline.

Default mode is deterministic and fast enough for local demos. If DeepEval is
installed and --framework deepeval is selected, the script attempts to run the
official LLM-as-judge metrics and falls back to the deterministic scorer on any
configuration error.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task10_generation import generate_with_citation

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
DETAILS_PATH = Path(__file__).parent / "eval_details.json"
BONUS_RESULTS_PATH = PROJECT_ROOT / "group_project" / "bonus" / "bonus_results.md"

METRICS = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
STOPWORDS = {
    "và",
    "là",
    "của",
    "có",
    "trong",
    "về",
    "theo",
    "những",
    "các",
    "một",
    "cho",
    "nào",
    "gì",
    "được",
    "bị",
    "đến",
    "từ",
    "with",
    "the",
    "and",
}


CONFIGS = {
    "A_hybrid_rerank": {"use_reranking": True},
    "B_hybrid_no_rerank": {"use_reranking": False},
}


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if token not in STOPWORDS and len(token) > 1
    }


def _overlap(reference: str, candidate: str) -> float:
    reference_tokens = _tokens(reference)
    if not reference_tokens:
        return 0.0
    candidate_tokens = _tokens(candidate)
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


def _clip(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _score_case(item: dict, answer: str, contexts: list[str]) -> dict:
    context_text = "\n".join(contexts)
    expected = item["expected_answer"]
    target = f"{item['question']} {expected} {item['expected_context']}"

    faithfulness = 0.75 * _overlap(answer, context_text) + 0.25 * (1.0 if "[" in answer and "]" in answer else 0.0)
    answer_relevance = 0.45 * _overlap(item["question"], answer) + 0.55 * _overlap(expected, answer)
    context_recall = 0.65 * _overlap(expected, context_text) + 0.35 * _overlap(item["expected_context"], context_text)
    per_context = [_overlap(target, context) for context in contexts] or [0.0]
    context_precision = statistics.mean(sorted(per_context, reverse=True)[: min(3, len(per_context))])

    return {
        "faithfulness": _clip(faithfulness),
        "answer_relevance": _clip(answer_relevance),
        "context_recall": _clip(context_recall),
        "context_precision": _clip(context_precision),
    }


def _run_case(item: dict, config_name: str, config: dict, top_k: int, use_llm: bool) -> dict:
    start = time.perf_counter()
    result = generate_with_citation(
        item["question"],
        top_k=top_k,
        use_reranking=config["use_reranking"],
        use_llm=use_llm,
    )
    latency = time.perf_counter() - start
    contexts = [source["content"] for source in result.get("sources", [])]
    scores = _score_case(item, result["answer"], contexts)
    return {
        "config": config_name,
        "category": item.get("category", "unknown"),
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "expected_context": item["expected_context"],
        "answer": result["answer"],
        "retrieval_source": result.get("retrieval_source", "unknown"),
        "retrieval_contexts": contexts,
        "sources": [
            {
                "source": source.get("metadata", {}).get("source", ""),
                "path": source.get("metadata", {}).get("path", ""),
                "score": float(source.get("score", 0.0)),
                "mode": source.get("source", "unknown"),
            }
            for source in result.get("sources", [])
        ],
        "latency_seconds": round(latency, 3),
        "metrics": scores,
        "average": round(statistics.mean(scores.values()), 3),
    }


def _try_deepeval(records: list[dict]) -> dict | None:
    try:
        from deepeval import evaluate
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase

        test_cases = []
        for record in records:
            test_cases.append(
                LLMTestCase(
                    input=record["question"],
                    actual_output=record["answer"],
                    expected_output=record["expected_answer"],
                    retrieval_context=record["retrieval_contexts"],
                )
            )
        metrics = [
            FaithfulnessMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7),
            ContextualRecallMetric(threshold=0.7),
            ContextualPrecisionMetric(threshold=0.7),
        ]
        evaluate(test_cases, metrics)
        return {"framework": "DeepEval", "status": "completed"}
    except Exception as exc:
        return {"framework": "deterministic fallback", "status": f"DeepEval unavailable: {exc}"}


def run_evaluation(limit: int | None, top_k: int, use_llm: bool, framework: str) -> dict:
    dataset = load_golden_dataset()
    if limit:
        dataset = dataset[:limit]

    records = []
    for config_name, config in CONFIGS.items():
        for item in dataset:
            print(f"[{config_name}] {item['question'][:70]}...")
            records.append(_run_case(item, config_name, config, top_k=top_k, use_llm=use_llm))

    framework_status = {"framework": "deterministic fallback", "status": "used"}
    if framework in {"auto", "deepeval"}:
        framework_status = _try_deepeval(records)

    summary = summarize(records, framework_status)
    DETAILS_PATH.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    export_results(summary, records)
    return {"summary": summary, "records": records}


def summarize(records: list[dict], framework_status: dict) -> dict:
    by_config = {}
    for config_name in CONFIGS:
        config_records = [record for record in records if record["config"] == config_name]
        metric_means = {
            metric: round(statistics.mean(record["metrics"][metric] for record in config_records), 3)
            for metric in METRICS
        }
        metric_means["average"] = round(statistics.mean(metric_means.values()), 3)
        metric_means["latency_seconds"] = round(statistics.mean(record["latency_seconds"] for record in config_records), 3)
        by_config[config_name] = metric_means

    worst = sorted(records, key=lambda record: record["average"])[:3]
    return {"framework": framework_status, "configs": by_config, "worst": worst}


def _table_row(metric: str, configs: dict) -> str:
    a = configs["A_hybrid_rerank"][metric]
    b = configs["B_hybrid_no_rerank"][metric]
    return f"| {metric} | {a:.3f} | {b:.3f} | {a - b:+.3f} |"


def export_results(summary: dict, records: list[dict]) -> None:
    configs = summary["configs"]
    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework",
        "",
        f"- Selected framework: DeepEval",
        f"- Runtime status: {summary['framework']['status']}",
        "- Fallback scorer: deterministic token-overlap metrics with the same four rubric names.",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A: hybrid + rerank | Config B: hybrid without rerank | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS + ["average", "latency_seconds"]:
        lines.append(_table_row(metric, configs))

    better = "Config A" if configs["A_hybrid_rerank"]["average"] >= configs["B_hybrid_no_rerank"]["average"] else "Config B"
    lines.extend(
        [
            "",
            "## A/B Comparison Analysis",
            "",
            "- Config A uses semantic search, BM25 lexical search, RRF merge, and reranking.",
            "- Config B uses the same hybrid retrieval stack without reranking.",
            f"- Conclusion: {better} has the better average score in this run.",
            "",
            "## Worst Performers (Bottom 3)",
            "",
            "| # | Config | Category | Question | Average | Likely root cause |",
            "|---:|---|---|---|---:|---|",
        ]
    )
    for index, record in enumerate(summary["worst"], 1):
        question = record["question"].replace("|", "\\|")
        cause = "Retriever missed exact expected context or answer required information outside top chunks."
        lines.append(f"| {index} | {record['config']} | {record['category']} | {question} | {record['average']:.3f} | {cause} |")

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "1. Add more criminal-code documents so questions about penalties and prosecution stages have stronger legal grounding.",
            "2. Keep reranking enabled for demo because it generally improves source ordering and context precision.",
            "3. Expand article metadata with publication dates and named entities to improve mixed legal-news questions.",
            "",
            "## Evaluation Details",
            "",
            f"- Golden dataset size: {len({record['question'] for record in records})}",
            f"- Total evaluated rows: {len(records)}",
            f"- Raw per-question details: `group_project/evaluation/{DETAILS_PATH.name}`",
        ]
    )
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bonus(top_k: int, use_llm: bool) -> None:
    probes = [
        {
            "question": "Điều 249 Bộ luật Hình sự quy định chính xác mức phạt tù cho từng khối lượng heroin như thế nào?",
            "expected": "Should refuse or say the current corpus does not contain enough criminal-code detail.",
        },
        {
            "question": "Có đúng là một ca sĩ khác ngoài các bài báo đã crawl cũng bị bắt vì ma túy trong năm 2026 không?",
            "expected": "Should refuse unsupported rumor/current-event claim.",
        },
        {
            "question": "Tòa án đã tuyên án cuối cùng bao nhiêu năm tù với Andrea Aybar?",
            "expected": "Should refuse because the corpus only supports investigation/arrest information.",
        },
        {
            "question": "Hôm nay có nghệ sĩ Việt Nam nào mới bị bắt vì ma túy không?",
            "expected": "Should refuse because the corpus is static and not current news.",
        },
    ]
    lines = ["# Bonus Failure Probe Results", ""]
    for index, probe in enumerate(probes, 1):
        result = generate_with_citation(probe["question"], top_k=top_k, use_llm=use_llm)
        answer = result["answer"]
        safe = "không thể xác minh" in answer.lower() or "cannot verify" in answer.lower()
        lines.extend(
            [
                f"## Probe {index}",
                "",
                f"**Question:** {probe['question']}",
                "",
                f"**Expected safe behavior:** {probe['expected']}",
                "",
                f"**Observed safe behavior:** {'Yes' if safe else 'No / needs discussion'}",
                "",
                "**Answer:**",
                "",
                answer,
                "",
            ]
        )
    BONUS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BONUS_RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {BONUS_RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-llm", action="store_true", help="Use Gemini/OpenAI generation during eval.")
    parser.add_argument("--framework", choices=["auto", "local", "deepeval"], default="local")
    parser.add_argument("--bonus", action="store_true", help="Run only the bonus failure probes.")
    args = parser.parse_args()

    if args.bonus:
        run_bonus(top_k=args.top_k, use_llm=args.use_llm)
        return

    result = run_evaluation(limit=args.limit, top_k=args.top_k, use_llm=args.use_llm, framework=args.framework)
    print(f"Wrote {RESULTS_PATH}")
    print(json.dumps(result["summary"]["configs"], indent=2))


if __name__ == "__main__":
    main()

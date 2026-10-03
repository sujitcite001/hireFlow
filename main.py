"""Command-line entry point for local resume management."""

import argparse
from pathlib import Path

from config import MAX_RESULTS, RESUMES_DIR, ensure_data_directories
from core.evaluator import evaluate_search_labels
from core.hybrid_indexer import search_resumes
from core.ingestion import list_resume_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local HireFlow resumes")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list available resume files")
    search_parser = commands.add_parser("search", help="search resume contents")
    search_parser.add_argument("query", help="keywords or a short role description")
    search_parser.add_argument("--limit", type=int, default=MAX_RESULTS)
    search_parser.add_argument(
        "--refresh-index", action="store_true", help="re-parse and rebuild the local resume index"
    )
    search_parser.add_argument(
        "--use-gemini", action="store_true", help="enable Gemini semantic search"
    )
    evaluate_parser = commands.add_parser(
        "evaluate", help="measure search ranking against a labeled JSONL file"
    )
    evaluate_parser.add_argument("labels", type=Path, help="JSONL queries and relevant candidate IDs")
    evaluate_parser.add_argument("--k", type=int, default=10)
    evaluate_parser.add_argument("--use-gemini", action="store_true")
    args = parser.parse_args()

    ensure_data_directories()
    if args.command == "list":
        files = list_resume_files(RESUMES_DIR)
        print("\n".join(str(path) for path in files) or "No resumes found.")
        return

    if args.command == "evaluate":
        metrics = evaluate_search_labels(
            args.labels,
            RESUMES_DIR,
            k=args.k,
            use_semantic=args.use_gemini,
        )
        print(f"Queries evaluated: {metrics['queries']}")
        print(f"Precision@{args.k}: {metrics['precision_at_k']:.3f}")
        print(f"Recall@{args.k}: {metrics['recall_at_k']:.3f}")
        print(f"MRR: {metrics['mrr']:.3f}")
        return

    results = search_resumes(
        args.query,
        RESUMES_DIR,
        limit=max(args.limit, 0),
        use_semantic=args.use_gemini,
        refresh_index=args.refresh_index,
    )
    for result in results:
        print(f"{result['score']:.3f}  {result['name']}\n{result['snippet']}\n")
    if not results:
        print("No matching resumes found.")


if __name__ == "__main__":
    main()

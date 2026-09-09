from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import Settings
from .dataset_builder import DatasetV0Builder
from .dataset_evaluator import DatasetBatchEvaluator
from .dataset_io import validate_dataset, write_jsonl
from .literature import MultiSourceLiteratureSearcher, OpenAccessDownloader
from .service import Hy3ScholarService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hy3scholar")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="解析 PDF 并构建 Evidence Database")
    ingest.add_argument("pdf", nargs="+", type=Path)

    run = sub.add_parser("run", help="生成综述并执行可信评估")
    run.add_argument("--workspace", required=True)
    run.add_argument("--question", required=True)
    run.add_argument("--top-k", type=int, default=12)
    run.add_argument("--no-refine", action="store_true")

    evaluate = sub.add_parser("evaluate", help="评估已有 Markdown 综述")
    evaluate.add_argument("--workspace", required=True)
    evaluate.add_argument("--question", required=True)
    evaluate.add_argument("--review", required=True, type=Path)

    literature = sub.add_parser(
        "literature-search", help="检索 OpenAlex/Crossref/arXiv 并可下载开放 PDF"
    )
    literature.add_argument("--query", required=True)
    literature.add_argument("--limit", type=int, default=10, choices=range(1, 21))
    literature.add_argument("--output", type=Path, default=Path("literature_results"))
    literature.add_argument("--download", action="store_true")

    dataset_build = sub.add_parser(
        "dataset-build", help="联网检索并构建待人工审核的 Dataset v0"
    )
    dataset_build.add_argument("--topic", required=True)
    dataset_build.add_argument("--output", type=Path, default=Path("dataset_v0"))
    dataset_build.add_argument("--papers", type=int, default=6)
    dataset_build.add_argument("--cases", type=int, default=12)
    dataset_build.add_argument("--no-hy3-web", action="store_true")

    dataset_validate = sub.add_parser(
        "dataset-validate", help="检查 Dataset 结构、证据 ID 和人工审核状态"
    )
    dataset_validate.add_argument("--dataset", required=True, type=Path)

    dataset_evaluate = sub.add_parser(
        "dataset-evaluate", help="在已审核 Dataset 上批量推理并输出 CSV/Markdown"
    )
    dataset_evaluate.add_argument("--dataset", required=True, type=Path)
    dataset_evaluate.add_argument(
        "--output", type=Path, default=Path("evaluation_results")
    )
    dataset_evaluate.add_argument(
        "--allow-pending",
        action="store_true",
        help="仅用于调试；正式结果必须使用人工 approved 样本",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = Hy3ScholarService(Settings())
    try:
        if args.command == "ingest":
            workspace = service.create_workspace(args.pdf)
            print(workspace.model_dump_json(indent=2))
        elif args.command == "run":
            result = service.run(
                args.workspace,
                args.question,
                top_k=args.top_k,
                refine=not args.no_refine,
            )
            print(result.model_dump_json(indent=2))
        elif args.command == "evaluate":
            result = service.evaluate(
                args.workspace,
                args.question,
                args.review.read_text(encoding="utf-8"),
            )
            print(result.model_dump_json(indent=2))
        elif args.command == "literature-search":
            searcher = MultiSourceLiteratureSearcher(service.settings)
            downloader = OpenAccessDownloader(service.settings)
            try:
                report = searcher.search(
                    [args.query], limit_per_source=min(args.limit, 10)
                )
                args.output.mkdir(parents=True, exist_ok=True)
                selected = report.records[: args.limit]
                write_jsonl(args.output / "search_results.jsonl", selected)
                if args.download:
                    manifests = [
                        downloader.download(record, args.output / "papers")
                        for record in selected
                        if record.is_open_access and record.pdf_url
                    ][: service.settings.max_downloads]
                    write_jsonl(args.output / "download_manifest.jsonl", manifests)
                print(report.model_copy(update={"records": selected}).model_dump_json(indent=2))
            finally:
                searcher.close()
                downloader.close()
        elif args.command == "dataset-build":
            builder = DatasetV0Builder(service.settings, service.client)
            try:
                report = builder.build(
                    topic=args.topic,
                    output_dir=args.output,
                    paper_count=args.papers,
                    case_count=args.cases,
                    use_hy3_web=not args.no_hy3_web,
                )
                print(report.model_dump_json(indent=2))
            finally:
                builder.close()
        elif args.command == "dataset-validate":
            report = validate_dataset(args.dataset)
            print(report.model_dump_json(indent=2))
            return 0 if report.valid else 2
        elif args.command == "dataset-evaluate":
            evaluator = DatasetBatchEvaluator(service.client, service.settings)
            results = evaluator.evaluate(
                args.dataset,
                args.output,
                allow_pending=args.allow_pending,
            )
            print(f"完成 {len(results)} 个样本；结果目录：{args.output.resolve()}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

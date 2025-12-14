from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m xgb_complex_features")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run an experiment grid from YAML config.")
    p_run.add_argument("--config", required=True, help="Path to YAML config.")

    p_agg = sub.add_parser("aggregate", help="Aggregate run-level results into summaries and deltas.")
    p_agg.add_argument("--input", required=True, help="Input run directory (contains results).")
    p_agg.add_argument("--output", required=True, help="Output directory for aggregated tables.")

    p_rep = sub.add_parser("report", help="Generate a Markdown report from aggregated outputs.")
    p_rep.add_argument("--input", required=True, help="Aggregated output directory.")
    p_rep.add_argument("--output", required=True, help="Path to report.md.")

    p_work = sub.add_parser("workflow", help="Run + aggregate + report for a single config.")
    p_work.add_argument("--config", required=True, help="Path to YAML config.")
    p_work.add_argument("--aggregate-output", help="Directory for aggregated tables (default: <run_dir>/aggregate).")
    p_work.add_argument("--report", help="Path to report.md (default: <aggregate_output>/report.md).")

    sub.add_parser("smoke", help="Run the end-to-end smoke suite (run+aggregate+report).")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "run":
        from xgb_complex_features.runner.execute import run_experiment

        run_experiment(config_path=args.config)
        return 0
    if args.command == "aggregate":
        from xgb_complex_features.reporting.aggregate import aggregate_runs

        aggregate_runs(input_dir=args.input, output_dir=args.output)
        return 0
    if args.command == "report":
        from xgb_complex_features.reporting.report_md import build_report

        build_report(input_dir=args.input, output_path=args.output)
        return 0
    if args.command == "workflow":
        from xgb_complex_features.runner.execute import run_experiment
        from xgb_complex_features.reporting.aggregate import aggregate_runs
        from xgb_complex_features.reporting.report_md import build_report

        run_dir = run_experiment(config_path=args.config)
        agg_dir = Path(args.aggregate_output) if args.aggregate_output else run_dir / "aggregate"
        aggregate_runs(input_dir=str(run_dir), output_dir=str(agg_dir))
        report_path = Path(args.report) if args.report else agg_dir / "report.md"
        build_report(input_dir=str(agg_dir), output_path=str(report_path))
        return 0
    if args.command == "smoke":
        from xgb_complex_features.runner.execute import run_experiment
        from xgb_complex_features.reporting.aggregate import aggregate_runs
        from xgb_complex_features.reporting.report_md import build_report

        run_experiment(config_path="configs/smoke.yaml")
        agg_dir = Path("runs/smoke/aggregate")
        aggregate_runs(input_dir="runs/smoke", output_dir=str(agg_dir))
        build_report(input_dir=str(agg_dir), output_path="runs/smoke/report.md")
        return 0

    raise AssertionError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

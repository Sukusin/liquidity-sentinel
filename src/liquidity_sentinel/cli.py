from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import run_backtest
from .dashboard import build_dashboard_html
from .pipeline import run_pipeline
from .sample_data import generate_sample_raw_data
from .utils import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RU Liquidity Sentinel")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-sample", help="Generate demo CSV data")
    gen.add_argument("--data-dir", type=Path, default=project_root() / "data" / "raw")

    run = sub.add_parser("run", help="Run full LSI pipeline")
    run.add_argument("--data-dir", type=Path, default=project_root() / "data" / "raw")
    run.add_argument("--output-dir", type=Path, default=project_root() / "output")
    run.add_argument("--train-until", default="2021-12-31")

    bt = sub.add_parser("backtest", help="Run backtest and sensitivity analysis")
    bt.add_argument("--data-dir", type=Path, default=project_root() / "data" / "raw")
    bt.add_argument("--output-dir", type=Path, default=project_root() / "output")
    bt.add_argument("--train-until", default="2021-12-31")

    dash = sub.add_parser("dashboard", help="Build static HTML dashboard")
    dash.add_argument("--output-dir", type=Path, default=project_root() / "output")
    dash.add_argument("--html-path", type=Path, default=project_root() / "output" / "dashboard.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate-sample":
        generate_sample_raw_data(args.data_dir)
        print(f"Sample data written to {args.data_dir}")
    elif args.command == "run":
        paths = run_pipeline(args.data_dir, args.output_dir, args.train_until)
        print("Pipeline finished:")
        for name, path in paths.items():
            print(f"- {name}: {path}")
    elif args.command == "backtest":
        paths = run_backtest(args.data_dir, args.output_dir, args.train_until)
        print("Backtest finished:")
        for name, path in paths.items():
            print(f"- {name}: {path}")
    elif args.command == "dashboard":
        path = build_dashboard_html(args.output_dir, args.html_path)
        print(f"Dashboard written to {path}")


if __name__ == "__main__":
    main()

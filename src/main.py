"""
Facade entry point. Constructs WfReq, dynamically loads the requested
pipeline's MainWf implementation, and calls execute().

Usage:
    python src/main.py <pipeline_name> [--start-from <step>] [--resume <checkpoint>]
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.common.wfs.dtos import WfReq, WfResp


def parse_args():
    parser = argparse.ArgumentParser(description="Run a pipeline")
    parser.add_argument("pipeline", help="Pipeline name (maps to src/pipelines/<name>/)")
    parser.add_argument("--start-from", dest="start_from", default=None,
                        help="Sub-workflow to start from")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint file to resume training")
    parser.add_argument("--config", default=None,
                        help="Override config path (default: configs/<pipeline>/default.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()

    req = WfReq(
        pipeline=args.pipeline,
        start_from=args.start_from,
        resume=args.resume,
        config_path=args.config,
    )

    module = importlib.import_module(f"pipelines.{args.pipeline}.main")
    pipeline_cls = getattr(module, "Pipeline")
    resp: WfResp = pipeline_cls().execute(req)

    if not resp.success:
        print(f"Pipeline failed: {resp.message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

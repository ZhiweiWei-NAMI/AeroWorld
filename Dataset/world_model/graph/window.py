"""Export a review window from the canonical Parquet label graph.

The structural vocabulary lives only in :mod:`Dataset.world_model.graph.contract`.
Window JSONL is an inspection view of an existing sample partition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Dataset.world_model.graph.contract import EDGE_ENDPOINTS, NODE_TYPES, REPO
from Dataset.world_model.graph.records import export_window_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    export_window_jsonl(args.partition, args.start, args.end, args.output, limit=args.limit)
    print(args.output)


if __name__ == "__main__":
    main()

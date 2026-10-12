"""Rebuild the current ARM index from validated UE delivery packages."""
from __future__ import annotations

import json

from Dataset.tools.arm_ue_export import ROOT, write_rows
from Dataset.tools.collect_arm_ue_delivery import collect


def main() -> int:
    delivery = collect()
    write_rows(ROOT / "arms/index.jsonl", delivery["packages"])
    print(json.dumps({key: delivery[key] for key in ("profile_count", "episode_count", "group_count", "arm_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

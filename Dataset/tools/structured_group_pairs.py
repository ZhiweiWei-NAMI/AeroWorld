"""Validate actual factual/alternative windows, independently of catalog counts."""

from collections import defaultdict
import json
import math
from typing import Any, Iterable, Mapping


IDENTITY_FIELDS = ("parent_episode_id", "group_id", "branch_group_id", "seed_id", "split")


def validate_window(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    for field in (*IDENTITY_FIELDS, "entity_id", "source_branch_root", "variant"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"window missing identity: {field}")
    if row["split"] not in {"train", "validation", "test"}:
        raise ValueError("invalid window split")
    origin, target = row.get("origin_tick"), row.get("target_tick")
    if type(origin) is not int or type(target) is not int or target != origin + 25:
        raise ValueError("invalid window time horizon")
    if origin % 5 or not 35 <= origin < target <= 900:
        raise ValueError("window is outside the formal capture grid")
    if row.get("history_frames") != 8 or row.get("future_frames") != 5:
        raise ValueError("window shape changed")
    if "branch_fork_tick" in row or "branch_window_end_tick" in row:
        fork, end = row.get("branch_fork_tick"), row.get("branch_window_end_tick")
        if type(fork) is not int or type(end) is not int or not fork <= origin < target <= end:
            raise ValueError("window exceeds its declared branch evidence")
    subject = row.get("state_subject")
    if subject is not None and (not isinstance(subject, Mapping)
                                or subject.get("entity_id") != row["entity_id"]
                                or not subject.get("source") or not subject.get("source_field")):
        raise ValueError("state subject provenance disagrees with window")
    for name, length in (("history_state", 8), ("future_states", 5)):
        values = row.get(name)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"invalid {name} shape")
        for state in values:
            if not isinstance(state, list) or len(state) != 7:
                raise ValueError(f"invalid {name} state width")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in state):
                raise ValueError(f"nonfinite or nonnumeric {name}")
    if row.get("state") != row["history_state"][-1] or row.get("next_state") != row["future_states"][0]:
        raise ValueError("state targets disagree with measured sequence")
    if row.get("predicate_value") not in {"true", "false", "unknown", "out_of_scope"}:
        raise ValueError("invalid predicate truth value")
    if row.get("predicate_mask") is not (row["predicate_value"] in {"true", "false"}):
        raise ValueError("predicate truth mask disagrees with known truth")
    return row["variant"], row["entity_id"], origin, target


def history_signature(row: Mapping[str, Any]) -> str:
    return json.dumps(
        row["history_state"], separators=(",", ":"), allow_nan=False
    )


def evaluate_group_pairs(group: Mapping[str, Any], windows: Iterable[Mapping[str, Any]]) -> dict:
    arms = {arm["arm_id"]: arm for arm in group["arms"]}
    factual = [arm for arm in arms.values() if arm.get("arm_kind") == "factual"]
    if len(factual) != 1:
        raise ValueError("group must have exactly one factual ARM")
    indexed = defaultdict(dict)
    split = None
    for row in windows:
        variant, entity, origin, target = validate_window(row)
        if variant not in arms:
            raise ValueError("window references an unknown ARM")
        arm = arms[variant]
        if row["parent_episode_id"] != group["episode_id"] or any(
            row[field] != group[field] for field in ("group_id", "branch_group_id", "seed_id")
        ):
            raise ValueError("window belongs to a different group")
        if row["source_branch_root"] != arm["branch_root"] or row.get("arm_kind") != arm["arm_kind"]:
            raise ValueError("window branch source or kind mismatch")
        if row.get("predicate_id") != group["target"]["predicate_id"]:
            raise ValueError("window changes the target predicate")
        if split is not None and split != row["split"]:
            raise ValueError("group crosses dataset splits")
        split = row["split"]
        key = (entity, origin, target)
        if key in indexed[variant]:
            raise ValueError("duplicate ARM window identity")
        indexed[variant][key] = row
    factual_id = factual[0]["arm_id"]
    pairs = []
    rejected = []
    for variant, arm in arms.items():
        if variant == factual_id:
            continue
        comparison = arm.get("comparison") or {}
        if comparison.get("status") != "PASS" or comparison.get("measured_change") is not True:
            rejected.append({"arm_id": variant, "reason": "MEASURED_ALTERNATIVE_NOT_VALIDATED"})
            continue
        common = sorted(set(indexed[factual_id]) & set(indexed[variant]))
        # Known truth is required for the pair's semantic target. Unknown and
        # out-of-scope windows remain inventory, not evidence of coverage.
        common = [key for key in common if indexed[factual_id][key]["predicate_mask"]
                  and indexed[variant][key]["predicate_mask"]]
        if not common:
            rejected.append({"arm_id": variant, "reason": "NO_COMMON_KNOWN_TARGET_WINDOW"})
            continue
        shared = [key for key in common if history_signature(indexed[factual_id][key])
                  == history_signature(indexed[variant][key])]
        key = shared[0] if shared else common[0]
        pairs.append({"factual_arm_id": factual_id, "alternative_arm_id": variant,
                      "entity_id": key[0], "origin_tick": key[1], "target_tick": key[2],
                      "comparable_window_count": len(common),
                      "identical_history_window_count": len(shared),
                      "window_pair_kind": "identical_history" if shared else "branch_rollout",
                      "factual_source": factual[0]["branch_root"],
                      "alternative_source": arm["branch_root"]})
    return {"episode_id": group["episode_id"], "group_id": group["group_id"],
            "branch_group_id": group["branch_group_id"], "split": split,
            "go_factual_count": int(bool(indexed[factual_id])),
            "go_alternative_count": len(pairs), "pairs": pairs, "rejected": rejected,
            "status": "GO_COMPLETE" if pairs else "NO_GO_PAIR_MISSING"}

"""Fixed L/P comparisons under the P01 mature-datagram producer contract."""
from __future__ import annotations

VERSION = "p01.ns3.v1"
CONTRACT_VERSION = "p01.ns3.mature-datagram-lp/v1"
RULES = (
    ("L", "communication.data_flow_latency_above_threshold", "latency_mean_ms", 120.0),
    ("P", "communication.packet_loss_above_threshold", "packet_loss_ratio", 0.05),
)


def bind_predicates(episode, network_states, compute_states, config):
    """Bind measured link operands only; local CPU states do not supply L/P.

    Existing rule names identify the physical comparisons. The producer contract
    explicitly distinguishes mature accepted datagrams from legacy operands.
    No legacy compute, cellular or MAC predicates are inferred by this slice.
    """
    rows = []
    for state in network_states:
        for symbol, predicate_id, field, threshold in RULES:
            metric = state[field]
            if state["status"] == "OUT_OF_SCOPE":
                value, reason = "out_of_scope", state["reason"]
            elif state["status"] != "IN_SCOPE":
                raise ValueError("undeclared network-state scope status")
            elif metric["status"] == "UNKNOWN":
                value, reason = "unknown", metric["reason"]
            elif metric["status"] == "KNOWN":
                value = "true" if metric["value"] > threshold else "false"
                reason = None
            else:
                raise ValueError("undeclared network metric availability status")
            rows.append({
                "episode_id": episode["episode_id"], "profile": config["profile"],
                "time_ns": state["time_ns"], "predicate_id": predicate_id,
                "symbol": symbol, "value": value, "reason": reason,
                "producer_version": VERSION, "producer_contract_version": CONTRACT_VERSION,
                "source_ref": "network_states.jsonl", "config_ref": "run_config.json",
                "link_id": state["link_id"], "flow_id": state["flow_id"],
                "src_node_id": state["src_node_id"], "dst_node_id": state["dst_node_id"],
                "owner": state["owner"], "life_epoch": state["life_epoch"],
                "receiver_owner": state["receiver_owner"],
                "receiver_life_epoch": state["receiver_life_epoch"],
                "scope_status": state["status"], "time_support": state["time_support"],
                "operand": {"field": field, **metric},
                "operator": "gt", "threshold": threshold,
                "unit": metric["unit"], "dimension": metric["dimension"],
            })
    return rows

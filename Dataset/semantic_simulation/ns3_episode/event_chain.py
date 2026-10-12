"""Owner-bound L6-2_v1 event-chain adapter: timed offered-load schedule and
measured response timeline.

Inputs, compiled exactly once per run: the named authored scenario script, the
frozen episode records under aw_data/render_ready_episodes/L6-2_v1__seed00
(episode_manifest.json, global_entity_roster.json, event_realization.jsonl),
one existing ns-3 product run (run_config.json, node_registry.jsonl[.gz],
predicates.jsonl[.gz]), and the two design contracts
design/p01_ns3/timed_intervention_l6_2.json (ROOT provider boundary) and
design/p01_ns3/authored_chain_snapshot.json (authored-grid snapshot).
Nothing else is opened. The compile phase reads each named input once; every
response value is then computed from the held rows. There is no repeated
provenance, schema, catalog, or warehouse scan in any query path.

Grid authority: the episode fires only on the 5-tick capture grid
(capture_grid_step=5 on every realization row; manifest event_tick_step=5), so
authored whole-tick chain anchors 492 and 572 snap to observed dispatches 495
and 575. Planned and observed are reported side by side and never conflated.

Window authority: the configured intervention window is the authored 40s..52s
(anomaly_tick=400, recovery_tick=520) exactly as preserved in the ROOT
p01.ns3.timed-intervention/v1 contract; it is configured, never silently
re-derived from the whole-tick chain span.

Interpretation authority: old frozen scripted motion is scenario-authored and
must never be called ns-3-rule-triggered. The response timeline is a measured
L/P rule surface only; the old R1 baseline carries no mission L/P true window,
so no rule-triggered reaction can be proven, and recovery stays UNKNOWN with an
explicit reason instead of a fabricated value.

Packet arithmetic: 1200B at 8Mbps is an exact 1.2ms integer-nanosecond period
(the same lattice law as provider_adapter._period_ns). The configured
[40s,52s) window therefore carries exactly 10,000 lattice points with the last
TX at 51.9988s; every scheduled TX is an actual lattice point
start_ns + k * period_ns.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

SCENARIO_ID = "L6-2_v1"
EPISODE_ID = "L6-2_v1__seed00"
MISSION_OWNER = "uav_digital_l6_2_v1"
STATION_OWNER = "tower_l6_2_v1"
TICK_NS = 100_000_000
CAPTURE_TICK_STEP = 5
PAYLOAD_BYTES = 1200
LOAD_BPS = 8_000_000
# Exact integer-nanosecond period: 1200B * 8 = 9600bit at 8Mbps is 1.2ms.
LOAD_PERIOD_NS = 1_200_000
TIMED_INTERVENTION_SCHEMA = "p01.ns3.timed-intervention/v1"
EVENT_CHAIN_SCHEMA = "aero_world.event_chain_scenario.v1"
PATTERN_ID = "event_chain.offered_load.event_window"

# Static truth surface: the fixed rule comparisons remain in rule_binding.
RULE_OPERANDS = {
    "L": {"field": "latency_mean_ms", "threshold": 120.0, "unit": "ms", "dimension": "time"},
    "P": {"field": "packet_loss_ratio", "threshold": 0.05, "unit": "1", "dimension": "dimensionless"},
}
AUTHORED_EVENT_IDS = (
    "lifecycle_takeoff_uav_digital_l6_2_v1", "uav_c2_precheck_patrol", "c2_degradation",
    "uav_delayed_response", "backup_link_lock", "nominal_c2_restore",
    "lifecycle_landing_uav_digital_l6_2_v1",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _open_suffix(path):
    """One exact open handle for a name-declared gzip or plain JSONL file."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _read_jsonl(path):
    """Read one named record file, once."""
    with _open_suffix(path) as stream:
        return [json.loads(line) for line in stream]


def load_scenario(script_path):
    """Bind one exact authored scenario script."""
    script = _read_json(script_path)
    _require(script.get("$schema") == "event_script_v1", "undeclared event script schema")
    _require(script.get("scenario_id") == SCENARIO_ID, "scenario_id must be the named L6-2_v1 contract")
    return script


def planned_chain(script):
    """Authored whole-tick fire anchors from the trigger graph.

    tick triggers anchor at their authored tick; event_fired_after anchors at
    the parent's planned fire tick plus the authored delay. These are plan
    values: the episode executes them on the 5-tick capture grid.
    """
    triggers = {row["trigger_id"]: row for row in script["triggers"]}
    planned = {}
    for event in script["events"]:
        trigger = triggers[event["trigger_ref"]]
        if trigger["type"] == "tick":
            anchor = int(trigger["tick"])
        elif trigger["type"] == "event_fired_after":
            parent = planned.get(trigger["event_id"])
            _require(parent is not None, f"{event['event_id']}: parent event not planned yet")
            anchor = parent["planned_fire_tick"] + int(trigger["delay_ticks"])
        else:
            raise ValueError(f"undeclared trigger type {trigger['type']!r}")
        _require("max_fire_count" in event and int(event["max_fire_count"]) == 1,
                 f"{event['event_id']}: adapter binds single-fire authored events only")
        planned[event["event_id"]] = {
            "event_id": event["event_id"], "planned_fire_tick": anchor,
            "trigger_id": trigger["trigger_id"], "trigger_type": trigger["type"],
            "planned_delay_ticks": (None if trigger["type"] == "tick"
                                    else int(trigger["delay_ticks"])),
            "priority": int(event["priority"]), "intent_stage": event["intent_stage"],
            "causal_predecessor_intent": event["causal_predecessor_intent"],
            "action_order": [action["action_id"] for action in event["actions"]]}
    _require(tuple(planned) == AUTHORED_EVENT_IDS, "authored event set changed")
    return planned


def configured_window(script):
    """The configured intervention window: authored 40s..52s, never re-derived."""
    parameters = script["parameters"]
    anomaly_tick = int(parameters["anomaly_tick"])
    recovery_tick = int(parameters["recovery_tick"])
    _require(anomaly_tick == 400 and recovery_tick == 520,
             "authored degradation parameters differ from the declared 40s/52s contract")
    payload, rate = int(PAYLOAD_BYTES), int(LOAD_BPS)
    period, remainder = divmod(payload * 8 * 1_000_000_000, rate)
    _require(remainder == 0, "offered load requires an exact integer-nanosecond period")
    _require(period == LOAD_PERIOD_NS, f"period {period} differs from the declared 1.2ms")
    start_ns = anomaly_tick * TICK_NS
    end_ns = recovery_tick * TICK_NS
    span_ns = end_ns - start_ns
    packets, leftover = divmod(span_ns, period)
    _require(leftover == 0, "configured window does not close on the 1.2ms lattice")
    return {
        "window_authority": "script parameters anomaly_tick/recovery_tick; ROOT traffic window preserved",
        "payload_bytes": payload, "offered_load_bps": rate, "period_ns": period,
        "start_tick": anomaly_tick, "end_tick": recovery_tick,
        "start_ns": start_ns, "end_ns": end_ns, "span_ns": span_ns,
        "scheduled_packets": packets,
        "last_scheduled_tx_ns": start_ns + (packets - 1) * period,
        "first_tx_ns": start_ns,
        "packet_identity_rule": "TX instances are exact lattice points start_ns + k*period_ns, k=0..N-1",
    }


def export_provider_boundary(window, source_ref, mechanism, assumptions):
    """The authoritative provider boundary, mirroring the preserved ROOT file."""
    return {
        "schema_version": TIMED_INTERVENTION_SCHEMA,
        "episode_id": EPISODE_ID,
        "traffic_windows": [
            {"flow_id": f"{EPISODE_ID}:c2_telemetry",
             "link_id": f"{EPISODE_ID}:c2_telemetry_link",
             "src_owner": MISSION_OWNER, "life_epoch": 0,
             "payload_bytes": 256, "period_ns": 100_000_000,
             "start_ns": 3_200_000_000, "end_ns": 90_000_000_000},
            {"flow_id": f"{EPISODE_ID}:scheduled_burst",
             "link_id": f"{EPISODE_ID}:scheduled_burst_link",
             "src_owner": MISSION_OWNER, "life_epoch": 0,
             "payload_bytes": window["payload_bytes"], "period_ns": window["period_ns"],
             "start_ns": window["start_ns"], "end_ns": window["end_ns"]},
        ],
        "source_ref": source_ref,
        "mechanism": mechanism,
        "assumptions": assumptions,
    }


def load_source_contract(design_path):
    """Read the preserved ROOT provider boundary once."""
    document = _read_json(design_path)
    _require(document["schema_version"] == TIMED_INTERVENTION_SCHEMA,
             "ROOT timed intervention contract version changed")
    _require(document["episode_id"] == EPISODE_ID, "ROOT contract episode identity changed")
    return document


def load_episode_identity(source_dir):
    """Bind the episode identity and its declared grid step."""
    manifest = _read_json(Path(source_dir) / "episode_manifest.json")
    _require(manifest["scenario_id"] == SCENARIO_ID,
             f"episode manifest scenario {manifest['scenario_id']!r} is not the named L6-2_v1 contract")
    _require(manifest["episode_id"] == EPISODE_ID, "unexpected episode identity")
    stations = [row["entity_id"] for row
                in _read_json(Path(source_dir) / "global_entity_roster.json")["entities"]
                if (row.get("semantic_scope") or {}).get("ontology_class_id") == "world:CommunicationStation"
                and "communication_state" in row]
    _require(stations == [STATION_OWNER], f"roster station set {stations} differs from the scripted station")
    return {"episode_id": manifest["episode_id"],
            "duration_ticks": int(manifest["duration_ticks"]),
            "duration_ns": int(manifest["duration_ticks"]) * TICK_NS,
            "station_owner": stations[0], "n_entities": int(manifest["n_entities"]),
            "manifest_ref": str(Path(source_dir) / "episode_manifest.json"),
            "roster_ref": str(Path(source_dir) / "global_entity_roster.json")}


def load_product_run(product_dir):
    """Bind one exact existing ns-3 product run and its own configuration."""
    product_dir = Path(product_dir)
    config = _read_json(product_dir / "run_config.json")
    _require(config["episode_id"] == EPISODE_ID, "product run episode is not the named seed00 episode")
    return {"dir": product_dir, "config": config, "profile": config["profile"],
            "artifact_files": config["artifact_files"],
            "run_config_ref": str(product_dir / "run_config.json")}


def observed_realizations(source_dir, script, planned):
    """Real measured realization rows, joined to the planned chain by event id.

    The episode grid is the 5-tick capture grid declared on the rows; observed
    dispatches are reported as measured. Old frozen scripted motion is
    everything strictly before the degradation dispatch and is never labeled
    rule-triggered.
    """
    rows = _read_jsonl(Path(source_dir) / "event_realization.jsonl")
    _require(len(rows) == len(AUTHORED_EVENT_IDS),
             "named episode must carry exactly the authored realization set")
    grid_steps = {row.get("capture_grid_step") for row in rows}
    _require(grid_steps == {CAPTURE_TICK_STEP},
             f"realization capture grid {grid_steps} differs from the declared {CAPTURE_TICK_STEP}-tick grid")
    by_event = {row["event_id"]: row for row in rows}
    _require(tuple(by_event) == AUTHORED_EVENT_IDS, "realization event order changed")
    degradation_dispatch = by_event["c2_degradation"]["dispatch_tick"]
    entries = []
    for event_id in AUTHORED_EVENT_IDS:
        row = by_event[event_id]
        plan = planned[event_id]
        move = [action for action in row["action_realizations"]
                if action["action_type"] == "move_entity"]
        entries.append({
            "event_id": event_id, "sequence_no": row["sequence_no"],
            "planned": plan,
            "observed_dispatch_tick": row["dispatch_tick"],
            "planned_vs_observed": ("on_grid" if row["dispatch_tick"] == plan["planned_fire_tick"]
                                    else "snapped_to_capture_grid"),
            "planned_to_observed_delta_ticks": row["dispatch_tick"] - plan["planned_fire_tick"],
            "result_tick": row["result_tick"], "evidence_tick": row["evidence_tick"],
            "first_motion_tick": min((int(action["first_motion_tick"]) for action in move
                                      if action.get("first_motion_tick") is not None), default=None),
            "first_capture_motion_tick": min((int(action["first_capture_motion_tick"]) for action in move
                                              if action.get("first_capture_motion_tick") is not None), default=None),
            "terminal_tick": max((int(action["terminal_tick"]) for action in move
                                  if action.get("terminal_tick") is not None), default=None),
            "terminal_enu_m": next((action["terminal_enu_m"] for action in move
                                    if action.get("terminal_enu_m")), None),
            "action_status": {action["action_id"]: action["status"]
                              for action in row["action_realizations"]},
            "phase": ("old_frozen_scripted_motion" if row["dispatch_tick"] < degradation_dispatch
                      else "post_degradation_scripted_schedule"),
            "rule_triggered": False,
            "rule_trigger_evidence": "none; scripted episode realization, not an ns-3 L/P rule response",
            "dwell_ticks": None, "hysteresis": None,
            "absent_authoring": {"dwell": "not authored in event_script_v1 actions",
                                 "hysteresis": "not authored in event_script_v1 actions"}})
    landing = by_event["lifecycle_landing_uav_digital_l6_2_v1"]
    landing_move = [action for action in landing["action_realizations"]
                    if action["action_type"] == "move_entity"][0]
    feasibility = next(action["terminal_feasibility"] for action in script["events"][-1]["actions"]
                       if "terminal_feasibility" in action)
    return {
        "rows": entries,
        "capture_grid_step": CAPTURE_TICK_STEP,
        "degradation_dispatch_tick": degradation_dispatch,
        "old_frozen_motion_before_tick": degradation_dispatch,
        "triggered_phase_note": ("scripted schedule at/after the degradation dispatch; it is authored "
                                 "causality, not an ns-3-rule-triggered response"),
        "interpretation": "existing authored realization; not ns-3-reactive motion",
        "landing": {
            "observed_dispatch_tick": landing["dispatch_tick"],
            "first_motion_tick": landing_move["first_motion_tick"],
            "first_capture_motion_tick": landing_move["first_capture_motion_tick"],
            "terminal_tick": landing_move["terminal_tick"],
            "terminal_enu_m": landing_move["terminal_enu_m"],
            "touchdown_dwell_ticks": feasibility["touchdown_dwell_ticks"],
            "touchdown_altitude_tolerance_m": feasibility["touchdown_altitude_tolerance_m"],
            "landing_reference_enu_m": feasibility["landing_reference_enu_m"],
            "result_tick": landing["result_tick"], "evidence_tick": landing["evidence_tick"],
            "terminal_result_required": landing["terminal_result_required"]},
        "source_refs": {
            "event_realization_ref": str(Path(source_dir) / "event_realization.jsonl"),
            "event_script_ref": "Dataset/scenarios/L6_digital_layer/failure/L6-2_v1/event_script.json"},
    }


def load_predicates(product_dir, artifact_files):
    """Read the fixed-rule predicate rows of the named run, once."""
    return _read_jsonl(Path(product_dir) / artifact_files["predicates.jsonl"])


def derive_response_timeline(predicates, window):
    """Measured-rule response timeline for the exact mission owner flow.

    Only the bound owner/life_epoch flow's in-scope L/P rows enter the
    timeline; the per-window value is the fixed comparison of the measured
    row, and the scheduled intervention never writes a label. UNKNOWN and
    OUT_OF_SCOPE remain separate four-valued states. A hold is measured
    consecutive in-scope windows with a known-true operand; recovery start
    would be the first known-false window after such a hold. With no true hold
    there is no measured response and no proven recovery, which is reported
    as UNKNOWN with its reason, never as false.
    """
    flow_rows = [row for row in predicates
                 if row["owner"] == MISSION_OWNER and row["life_epoch"] == 0]
    _require(flow_rows, "named product run has no predicate row for the mission owner epoch")
    by_time = {}
    for row in flow_rows:
        if row["scope_status"] == "IN_SCOPE":
            by_time.setdefault(row["time_ns"], {})[row["symbol"]] = row
    timeline, true_hold, recovery = [], [], None
    truth_counts = {"L": {"true": 0, "false": 0, "unknown": 0},
                    "P": {"true": 0, "false": 0, "unknown": 0}}
    for time in sorted(by_time):
        pair = by_time[time]
        _require(set(pair) == {"L", "P"}, f"time {time}: in-scope window lacks the fixed L/P pair")
        measured = {}
        for symbol, row in pair.items():
            truth_counts[symbol][row["value"]] += 1
            operand = row["operand"]
            _require(operand["field"] == RULE_OPERANDS[symbol]["field"],
                     f"stored predicate {row['predicate_id']} is not the declared fixed-rule operand")
            measured[symbol] = {"value": row["value"], "reason": row["reason"],
                                "field": operand["field"], "operator": row["operator"],
                                "threshold": row["threshold"], "unit": operand["unit"],
                                "raw_value": operand.get("value"), "raw_status": operand.get("status")}
        known_true = any(pair[symbol]["value"] == "true" for symbol in ("L", "P"))
        known_false = all(pair[symbol]["value"] == "false" for symbol in ("L", "P"))
        if known_true:
            true_hold.append(time)
        if known_false and true_hold:
            if recovery is None:
                recovery = time
            true_hold = []
        timeline.append({
            "time_ns": time,
            "in_configured_intervention_window": window["start_ns"] <= time < window["end_ns"],
            "L": measured["L"], "P": measured["P"],
            "owner": MISSION_OWNER, "life_epoch": 0,
            "flow_id": pair["L"]["flow_id"], "link_id": pair["L"]["link_id"],
            "receiver_owner": pair["L"]["receiver_owner"],
            "receiver_life_epoch": pair["L"]["receiver_life_epoch"],
            "scope_status": pair["L"]["scope_status"], "time_support": pair["L"]["time_support"],
            "source_ref": pair["L"]["source_ref"], "config_ref": pair["L"]["config_ref"]})
    return {
        "owner": MISSION_OWNER, "life_epoch": 0,
        "flow_id": flow_rows[0]["flow_id"], "link_id": flow_rows[0]["link_id"],
        "in_scope_times": len(by_time),
        "out_of_scope_times": len(flow_rows) // 2 - len(by_time),
        "truth_counts": truth_counts,
        "true_hold": {"start_ns": true_hold[0] if true_hold else None,
                      "end_ns": true_hold[-1] if true_hold else None,
                      "count": len(true_hold)},
        "recovery_start_ns": recovery,
        "recovery_status": "measured" if recovery is not None else "UNKNOWN",
        "recovery_unknown_reason": (None if recovery is not None else
            "no measured known-true L/P hold in the old R1 baseline; no rule-triggered "
            "reaction can be proven from this run"),
        "rule_triggered_response": {
            "proven": recovery is not None,
            "basis": "measured L/P rule rows only; scheduled interventions never write labels",
            "evidence": {"known_true_windows": sum(1 for time in true_hold) if true_hold else 0,
                         "first_true_ns": true_hold[0] if true_hold else None}},
        "timeline": timeline}


def build_scenario_document(script, identity, planned, window, provider_boundary,
                            realizations, timeline, product):
    """Compact scenario JSON: declared schedule plus measured execution facts."""
    return {
        "schema": EVENT_CHAIN_SCHEMA,
        "scenario_id": SCENARIO_ID,
        "description": script["description"],
        "owners": {
            "mission_owner": MISSION_OWNER, "mission_life_epoch": 0,
            "station_owner": STATION_OWNER, "station_life_epoch": 0,
            "direction": f"{MISSION_OWNER} -> {STATION_OWNER}",
            "authority": "event_script action entity_ids and roster CommunicationStation"},
        "authored_chain": sorted(planned.values(), key=lambda row: row["planned_fire_tick"]),
        "configured_intervention_window": window,
        "provider_boundary": provider_boundary,
        "measurement_contract": {
            "predicate_source_ref": str(product["dir"] / product["artifact_files"]["predicates.jsonl"]),
            "rule_operands": RULE_OPERANDS,
            "profile": product["profile"],
            "run_config_ref": product["run_config_ref"],
            "epoch_birth_ns": 3_200_000_000,
            "epoch_birth_authority": "takeoff trigger tick 32; planned chain, matched by the registry epoch",
            "grid_rule": ("the episode fires on the 5-tick capture grid; planned whole-tick anchors "
                          "492/572 are observed at 495/575; planned and observed stay separate"),
            "capture_grid_step": CAPTURE_TICK_STEP},
        "chain_realization": {
            "degradation_dispatch_tick": realizations["degradation_dispatch_tick"],
            "old_frozen_motion_before_tick": realizations["old_frozen_motion_before_tick"],
            "phase_note": realizations["triggered_phase_note"],
            "interpretation": realizations["interpretation"],
            "rows": realizations["rows"],
            "landing": realizations["landing"],
            "dwell_hysteresis_authoring": "not authored; reported absent, never invented"},
        "response_timeline_summary": {
            "owner": timeline["owner"], "life_epoch": timeline["life_epoch"],
            "flow_id": timeline["flow_id"], "link_id": timeline["link_id"],
            "in_scope_times": timeline["in_scope_times"],
            "out_of_scope_times": timeline["out_of_scope_times"],
            "truth_counts": timeline["truth_counts"],
            "true_hold": timeline["true_hold"],
            "recovery_start_ns": timeline["recovery_start_ns"],
            "recovery_status": timeline["recovery_status"],
            "recovery_unknown_reason": timeline["recovery_unknown_reason"],
            "rule_triggered_response": timeline["rule_triggered_response"]},
        "identity": {"episode_id": identity["episode_id"],
                     "duration_ticks": identity["duration_ticks"],
                     "manifest_ref": identity["manifest_ref"]},
        "source_refs": {
            "event_script_ref": realizations["source_refs"]["event_script_ref"],
            "timed_intervention_root_ref": "design/p01_ns3/timed_intervention_l6_2.json",
            "authored_chain_snapshot_ref": "design/p01_ns3/authored_chain_snapshot.json",
            **realizations["source_refs"]},
    }


def run_once(scenario_path, source_dir, product_dir, design_dir, output_path):
    """Compile the named inputs once, calculate, persist, and report counts."""
    script = load_scenario(scenario_path)
    identity = load_episode_identity(source_dir)
    product = load_product_run(product_dir)
    root_contract = load_source_contract(Path(design_dir) / "timed_intervention_l6_2.json")
    planned = planned_chain(script)
    window = configured_window(script)
    _require(root_contract["episode_id"] == identity["episode_id"],
             "ROOT contract and episode identity differ")
    burst = next(row for row in root_contract["traffic_windows"] if row["payload_bytes"] == PAYLOAD_BYTES)
    _require((burst["start_ns"], burst["end_ns"], burst["period_ns"], burst["payload_bytes"])
             == (window["start_ns"], window["end_ns"], window["period_ns"], window["payload_bytes"]),
             "configured window differs from the preserved ROOT traffic window")
    realizations = observed_realizations(source_dir, script, planned)
    predicates = load_predicates(product["dir"], product["artifact_files"])
    timeline = derive_response_timeline(predicates, window)
    provider_boundary = export_provider_boundary(
        window, root_contract["source_ref"], root_contract["mechanism"], root_contract["assumptions"])
    document = build_scenario_document(script, identity, planned, window, provider_boundary,
                                       realizations, timeline, product)
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    return {
        "output_path": str(output),
        "scenario_id": SCENARIO_ID, "episode_id": identity["episode_id"],
        "owner": MISSION_OWNER, "life_epoch": 0, "station_owner": STATION_OWNER,
        "packet_arithmetic": {
            "payload_bytes": window["payload_bytes"], "offered_load_bps": window["offered_load_bps"],
            "period_ns": window["period_ns"], "window_start_ns": window["start_ns"],
            "window_end_ns": window["end_ns"], "window_span_ns": window["span_ns"],
            "scheduled_packets": window["scheduled_packets"],
            "last_scheduled_tx_ns": window["last_scheduled_tx_ns"],
            "last_scheduled_tx_tick_s": window["last_scheduled_tx_ns"] / 1e9},
        "planned_vs_observed": [
            {"event_id": row["event_id"], "planned_fire_tick": row["planned"]["planned_fire_tick"],
             "observed_dispatch_tick": row["observed_dispatch_tick"],
             "relation": row["planned_vs_observed"],
             "delta_ticks": row["planned_to_observed_delta_ticks"]}
            for row in realizations["rows"]],
        "predicate_rows_read": len(predicates),
        "timeline_windows": timeline["in_scope_times"],
        "timeline_out_of_scope_times": timeline["out_of_scope_times"],
        "timeline_truth_counts": timeline["truth_counts"],
        "true_hold": timeline["true_hold"],
        "recovery_status": timeline["recovery_status"],
        "recovery_start_ns": timeline["recovery_start_ns"],
        "rule_triggered_response_proven": timeline["rule_triggered_response"]["proven"],
        "old_motion_classification": realizations["rows"][0]["phase"],
        "landing_terminal_tick": realizations["landing"]["terminal_tick"],
        "landing_first_motion_tick": realizations["landing"]["first_motion_tick"],
        "run_config_ref": product["run_config_ref"],
        "root_contract_ref": "design/p01_ns3/timed_intervention_l6_2.json",
        "hot_path": "each named input opened once in compile; all response values computed from held rows",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path,
                        default=Path("Dataset/scenarios/L6_digital_layer/failure/L6-2_v1/event_script.json"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("aw_data/render_ready_episodes/L6-2_v1__seed00"))
    parser.add_argument("--product-dir", type=Path,
                        default=Path("aw_data/ns3_episode_v5_radio_reference/R1/L6-2_v1__seed00/baseline/run1"))
    parser.add_argument("--design-dir", type=Path, default=Path("design/p01_ns3"))
    parser.add_argument("--output", type=Path, default=Path("design/p01_ns3/event_chain.json"))
    arguments = parser.parse_args()
    result = run_once(arguments.scenario, arguments.source_dir, arguments.product_dir,
                      arguments.design_dir, arguments.output)
    print(json.dumps(result, allow_nan=False, indent=2))

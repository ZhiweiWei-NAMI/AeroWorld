"""Connect frozen core ledgers to scoped, executable P09 source products.

The products are side tables. They do not change the P01 import contract, adopt
candidate compute workloads, or project factual labels into intervention arms.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

VERSION = "p09.core-sources/v1"
GAP_CLASSES = {"source_unrecorded", "not_applicable", "wiring_missing",
               "conversion_error", "computation_missing", "observer_unseen"}


def json_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_lifetime(task: Mapping[str, Any], segment_start_s: float, tick_hz: int) -> dict[str, Any]:
    """Exact city/episode clocks and explicitly defined discrete source grid."""
    start, end = Decimal(str(task["start_s"])), Decimal(str(task["end_s"]))
    hold, origin = Decimal(str(task["terminal_hold_s"])), Decimal(str(segment_start_s))
    if not all(v.is_finite() for v in (start, end, hold, origin)) or end < start or hold < 0:
        raise ValueError(f"{task['task_id']}: malformed source lifetime")
    if type(tick_hz) is not int or tick_hz <= 0 or type(task["looping"]) is not bool:
        raise ValueError("malformed global source time grid or looping flag")
    inclusive = task["looping"] or hold > 0
    last = (end + hold - origin) * tick_hz
    return {"task_id": task["task_id"], "world_birth_s": float(start),
            "world_task_end_s": float(end), "world_presence_end_s": float(end+hold),
            "episode_birth_s": float(start-origin), "episode_task_end_s": float(end-origin),
            "episode_presence_end_s": float(end+hold-origin),
            "first_active_grid_tick": int(((start-origin)*tick_hz).to_integral_value(rounding=ROUND_CEILING)),
            "last_active_grid_tick": int(last.to_integral_value(rounding=ROUND_FLOOR)) if inclusive else int(last.to_integral_value(rounding=ROUND_CEILING))-1,
            "presence_end_inclusive": inclusive, "tick_hz": tick_hz,
            "segment_start_s": float(origin),
            "grid_meaning": "first_and_last_active_discrete_source_ticks_not_rounded_birth"}


def energy_step(soc: float, params: Mapping[str, Any], speed: float,
                temperature: float, elapsed_ticks: float, charged_ticks: float) -> dict[str, float]:
    """The retained payload_energy equation, with no new model parameters."""
    derating = min(float(params["max_power_derating_ratio"]), max(0.0,
        (temperature-float(params["derating_start_temperature_c"]))*float(params["temperature_to_derating_ratio"])))
    speed_factor = 1.0 + min(float(params["maximum_speed_energy_factor"]),
                            max(0.0, speed)*float(params["speed_energy_factor_per_mps"]))
    thermal_factor = 1.0 + min(float(params["maximum_derating_energy_factor"]),
                             derating*float(params["derating_energy_factor_per_ratio"]))
    consumed = float(params["soc_consumption_per_tick"])*elapsed_ticks*speed_factor*thermal_factor
    charged = float(params["soc_charge_per_tick"])*charged_ticks
    next_soc = min(1.0, max(float(params["minimum_soc"]), soc-consumed+charged))
    scale = max(float(params["minimum_range_scale"]),
                1.0-derating*float(params["derating_to_range_loss_factor"]))
    return {"state_of_charge_ratio": next_soc, "energy_consumed_ratio": consumed,
            "energy_charged_ratio": charged, "power_derating_ratio": derating,
            "predicted_range_m": float(params["nominal_range_m"])*next_soc*scale,
            "temperature_energy_factor": thermal_factor}


def world_energy_history(*, task: Mapping[str, Any], lifetime: Mapping[str, Any],
                         weather: Mapping[int, Mapping[str, Any]], ticks: Iterable[int],
                         params: Mapping[str, Any], sampler: Any,
                         charged_subject: bool | None, sample_step: int) -> tuple[dict[int, dict[str, Any]], str | None]:
    """Compute current-episode model history; never invent preceding weather.

    As in domain_state, each interval uses its right endpoint's sampled weather
    and speed. Exact fractional birth contributes its exact fractional duration
    in tick units; it is never rounded to a fictional integer birth tick.
    """
    birth = float(lifetime["episode_birth_s"])
    if birth < 0:
        return {}, "pre_episode_temperature_charging_and_initial_energy_history_unrecorded"
    if charged_subject is None:
        return {}, "complete_episode_charging_service_plan_unrecorded"
    if charged_subject:
        return {}, "charging_subject_requires_full_contact_and_operational_state_history"
    wanted = sorted(set(ticks))
    if not wanted:
        return {}, None
    hz = int(lifetime["tick_hz"])
    previous = birth*hz
    soc = float(params["initial_soc_ratio"])
    result = {}
    first_sample = int(math.ceil(previous/sample_step))*sample_step
    consumed_total = 0.0
    for tick in range(first_sample, wanted[-1]+1, sample_step):
        if tick > int(lifetime["last_active_grid_tick"]):
            break
        row = weather.get(tick)
        if row is None or type(row.get("temperature_c")) not in (int, float):
            return result, f"temperature_history_missing_at_tick:{tick}"
        absolute = float(lifetime["segment_start_s"])+tick/hz
        if not sampler.active_at(absolute):
            return result, f"task_motion_history_missing_at_tick:{tick}"
        position, yaw, speed = sampler.sample(absolute)
        step = energy_step(soc, params, float(speed), float(row["temperature_c"]), tick-previous, 0.0)
        consumed_total += step["energy_consumed_ratio"]
        soc = step["state_of_charge_ratio"]
        previous = tick
        if tick in wanted:
            route_distance = float(task["route_length_m"])
            result[tick] = {**step, "range_insufficient": step["predicted_range_m"] < route_distance+float(params["required_route_reserve_m"]),
                "planned_route_distance_m": route_distance, "speed_mps": speed,
                "temperature_c": row["temperature_c"], "modeled_charging_active": False,
                "charging_basis": "not_a_subject_of_the_complete_episode_charging_service_plan",
                "consumption_since_world_birth_ratio": consumed_total,
                "exact_birth_episode_s": birth, "integration_interval_start_tick_units": birth*hz,
                "integration_policy": "existing_right_endpoint_model_with_exact_partial_birth_duration",
                "position_enu_m": position, "yaw_deg": yaw}
    return result, None


def read_task_source(source: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Consume one explicit regime; preserve missing task outcomes and binding."""
    path = Path(source["tasks_path"])
    episode = source["episode_id"]
    for row in json_rows(path):
        owner = row["owner"]
        if owner["episode_id"] != episode or owner["entity_id"] not in source["owner_entity_ids"]:
            raise ValueError(f"{path}: task owner does not match source binding")
        completion, deadline = row.get("completion_ns"), row.get("deadline_ns")
        if type(completion) is int and type(deadline) is int:
            actual = completion > deadline
            if type(row.get("deadline_missed")) is bool and actual != row["deadline_missed"]:
                raise ValueError(f"{path}: deadline outcome differs from actual completion")
            basis = "actual_completion_ns_greater_than_deadline_ns"
        else:
            actual = row.get("deadline_missed") if type(row.get("deadline_missed")) is bool else None
            basis = "producer_terminal_event" if actual is not None else "terminal_outcome_unrecorded_or_censored"
        yield {"schema_version": VERSION, "source_id": source["source_id"],
               "applicability": source["applicability"], "regime": source["regime"],
               "episode_id": episode, "entity_id": owner["entity_id"],
               "owner_binding_status": source["owner_binding_status"][owner["entity_id"]],
               "task_id": row["task_id"], "actor_epoch": owner["actor_epoch"],
               "time_unit": "ns", "clock": "episode_simulation_time",
               "arrival_ns": row["arrival_ns"], "deadline_ns": deadline,
               "completion_ns": completion, "terminal_ns": row.get("terminal_ns"),
               "status": row["status"], "deadline_missed": actual,
               "deadline_source_status": "available" if actual is not None else "not_applicable" if row["deadline_status"] == "OUT_OF_SCOPE" else "source_unrecorded",
               "deadline_source_reason": row["deadline_reason"],
               "deadline_basis": basis, "cycles": row["cycles"],
               "memory_bytes": row["memory_bytes"], "source_class": "authored_deterministic_resource_simulation",
               "source_ref": str(path), "configuration_ref": source["config_path"]}


def domain_global_energy(inputs: Any, profile: Mapping[str, Any]) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, str], dict[str, dict[str, Any]]]:
    """The same source computation consumed by typed domain payload truth."""
    from Dataset.tools.uav_global_flow.generate_uav_flow import RouteSampler, UavTask
    global_entities = {key:entity for key,entity in inputs.roster_entities.items() if isinstance(entity.get("uav_global_flow"), dict)}
    if not global_entities:
        return {}, {}, {}
    declared = inputs.manifest_projection["uav_global_flow"]["source"]
    plan = _json(Path(declared["task_plan"]))
    manifest = _json(Path(declared["manifest"]))
    tasks_by_uav = {task["uav_id"]:task for task in plan["tasks"]}
    if len(tasks_by_uav) != len(plan["tasks"]):
        raise ValueError("global task plan has ambiguous UAV ownership")
    task_fields = {field.name for field in fields(UavTask)}
    charged_subjects = {request["uav_id"] for facility in inputs.charging_service_plan["facilities"] for request in facility["requests"]}
    checkpoints, gaps, lifetimes = {}, {}, {}
    ticks = [frame["tick"] for frame in inputs.frames]
    for entity_id, entity in global_entities.items():
        # Strict objective inputs remove authored task_id labels. Bind the
        # physical UAV to its unique task in the declared simulator source.
        task = tasks_by_uav[entity["uav_id"]]
        if task["uav_id"] != entity["uav_id"]:
            raise ValueError(f"{inputs.episode_id}:{entity_id}: task owner differs")
        segment_start = entity["uav_segment"]["segment_start_s"]
        life = task_lifetime(task, segment_start, manifest["tick_hz"])
        lifetimes[entity_id] = life
        history,gap = world_energy_history(task=task,lifetime=life,weather=inputs.weather_by_tick,ticks=ticks,
            params=profile["domain_models"]["payload_energy"],
            sampler=RouteSampler(UavTask(**{k:v for k,v in task.items() if k in task_fields})),
            charged_subject=entity_id in charged_subjects,sample_step=profile["authoritative_tick_policy"]["step"])
        checkpoints[entity_id] = history
        if gap is not None:
            gaps[entity_id] = gap
    return checkpoints,gaps,lifetimes


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def declared_arm_scope(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Read the explicit scopes of the current layer contracts.

    L2 and L5 declare predicate lists at the manifest root; L3 declares a
    primary predicate. The L3 primary claim is never expanded to a whole bank.
    """
    if arm.get("schema") == "aero_l6_v2_arm_manifest":
        return {"entity_ids":arm["scope"]["entity_ids"],"predicate_ids":arm["predicate_ids"],
                "scope_authority":"aero_l6_v2_arm_manifest","scope_meaning":"scope_entities_and_explicit_root_predicate_list"}
    if "scope" in arm:
        return dict(arm["scope"])
    schema = arm.get("schema_name") if "schema_name" in arm else arm["schema"]
    if schema in {"l2_event_chain_window_arm", "aeroworld_l5_candidate_window"}:
        return {"entity_ids":arm["entity_ids"],"predicate_ids":arm["predicate_ids"],
                "scope_authority":schema,"scope_meaning":"explicit_manifest_entity_and_predicate_lists"}
    if schema == "aeroworld_l3_v2_arm":
        return {"entity_ids":arm["entity_ids"],"predicate_ids":[arm["primary_predicate"]],
                "scope_authority":schema,"scope_meaning":"declared_primary_predicate_only"}
    raise ValueError(f"Unsupported current ARM scope contract: {schema}")


def produce(project: Path, output: Path, *, compute_sources: list[tuple[Path,str,str]], episode_filter: str | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(project))
    from Dataset.tools.uav_global_flow.generate_uav_flow import RouteSampler, UavTask
    integration = _load_module(Path(__file__).resolve().parents[1]/"tools/uav_global_flow/truth_integration.py", "p09_source_integration")
    source_root = project/"aw_data/uav_outputs/donghu_uav_flow_270s"
    dataset = integration.UavGlobalFlowDataset(source_root)
    task_fields = {field.name for field in fields(UavTask)}
    samplers = {key: RouteSampler(UavTask(**{k:v for k,v in task.items() if k in task_fields}))
                for key, task in dataset.tasks_by_id.items()}
    profile_path = project/"Dataset/semantic_rules/profiles/domain_state_supplement_profile.json"
    profile = _json(profile_path)
    params = profile["domain_models"]["payload_energy"]
    output.mkdir(parents=True, exist_ok=True)
    counts, status_counts = Counter(), Counter()
    episode_dirs = sorted((project/"aw_data/render_ready_episodes_capture_filtered").glob("*__seed*"))
    if episode_filter:
        episode_dirs = [p for p in episode_dirs if p.name == episode_filter]
    with (output/"global_uav_lifetimes.jsonl").open("w") as lifetime_stream, (output/"global_uav_energy.jsonl").open("w") as energy_stream, (output/"core_field_status.jsonl").open("w") as status_stream:
        def emit_status(row: dict[str, Any]) -> None:
            status_stream.write(json.dumps({"schema_version":VERSION, **row})+"\n")
            status_counts[row["status"]] += 1
        for episode_dir in episode_dirs:
            episode = episode_dir.name
            manifest = _json(episode_dir/"episode_manifest.json")
            roster = _json(episode_dir/"global_entity_roster.json")["entities"]
            globals_by_id = {row["entity_id"]: row for row in roster if isinstance(row.get("uav_global_flow"), dict)}
            segment = dataset.segment_for_episode(episode, manifest)
            battery_path = project/"aw_data/objective_semantic_truth"/episode/"simulation_logs/battery_log.jsonl"
            observations: dict[str, list[tuple[int, Mapping[str, Any]]]] = {key:[] for key in globals_by_id}
            existing_energy_status: dict[str, Counter] = {}
            existing_energy_missing: dict[str, set[str]] = {}
            if battery_path.is_file():
                for row in json_rows(battery_path):
                    entity = row["scope_entity_id"]
                    values = row["state"]["values"]
                    if entity not in existing_energy_status:
                        existing_energy_status[entity] = Counter()
                        existing_energy_missing[entity] = set()
                    counter = existing_energy_status[entity]
                    counter["known_soc" if type(values.get("state_of_charge_ratio")) in (int,float) else "unresolved_soc"] += 1
                    if values.get("range_insufficient") is False: counter["actual_range_false"] += 1
                    elif values.get("range_insufficient") is True: counter["actual_range_true"] += 1
                    existing_energy_missing[entity].update(values["missing_inputs"])
                    if entity in observations:
                        observations[entity].append((row["tick"],values))
            for entity_id,counter in existing_energy_status.items():
                reasons = sorted(existing_energy_missing[entity_id])
                status = "available" if not counter["unresolved_soc"] else "wiring_missing" if any("activation_tick" in value for value in reasons) else "source_unrecorded" if any("declared_source_artifact_missing" in value or "source_episode_dir_missing" in value for value in reasons) else "computation_missing"
                emit_status({"episode_id":episode,"entity_id":entity_id,"field":"existing_frozen_payload_energy",
                    "status":status,"measured_counts":dict(counter),"missing_inputs":reasons,
                    "source_ref":str(battery_path),"source_class":"declared_model_observation_not_battery_telemetry",
                    "frozen_ledger_regenerated":False,"resolved_by":"global_uav_energy.jsonl" if entity_id in globals_by_id else None})
            weather = {row["tick"]:row for row in json_rows(episode_dir/"weather_meta.jsonl")}
            charging_path = project/"aw_data/charging_supplement"/episode/"charging_service_plan.json"
            if charging_path.is_file():
                charging_plan = _json(charging_path)
                charging_subjects = {request["uav_id"] for facility in charging_plan["facilities"] for request in facility["requests"]}
            else:
                charging_subjects = None
                emit_status({"episode_id":episode,"field":"charging_service_plan","status":"source_unrecorded","source_ref":str(charging_path),"energy_computation_permitted":False})
            if not battery_path.is_file():
                emit_status({"episode_id":episode,"field":"existing_frozen_payload_energy","status":"computation_missing","source_ref":str(battery_path),"observer_visibility_claimed":False})
            for entity_id, entity in globals_by_id.items():
                task = dataset.tasks_by_id[entity["task_id"]]
                if task["uav_id"] != entity["uav_id"]:
                    raise ValueError(f"{episode}:{entity_id}: task/UAV identity divergence")
                life = dataset.task_lifetime(entity["task_id"], segment)
                ticks = sorted({tick for tick,_ in observations[entity_id]})
                life_row = {"schema_version":VERSION,"episode_id":episode,"entity_id":entity_id,"uav_id":entity["uav_id"],**life,
                            "first_observed_tick":min(ticks) if ticks else None,"last_observed_tick":max(ticks) if ticks else None,
                            "observed_sample_ticks":ticks,"visibility_basis":"actual_frozen_battery_observation_ticks",
                            "capture_visibility_claimed":False,"world_observation_distinct_from_sensor_visibility":True,
                            "observation_source_ref":str(battery_path),"source_class":"declared_simulator_task_plan"}
                lifetime_stream.write(json.dumps(life_row)+"\n")
                counts["global_lifetimes"] += 1
                histories, gap = world_energy_history(task=task,lifetime=life,weather=weather,ticks=ticks,params=params,
                    sampler=samplers[entity["task_id"]],charged_subject=entity_id in charging_subjects if charging_subjects is not None else None,
                    sample_step=profile["authoritative_tick_policy"]["step"])
                for tick, existing in observations[entity_id]:
                    known = type(existing.get("state_of_charge_ratio")) in (int,float)
                    if known:
                        values, origin = dict(existing), "existing_frozen_model_observation_preserved"
                        counts["existing_energy_preserved"] += 1
                    elif tick in histories:
                        values, origin = histories[tick], "actual_current_episode_source_history_computation"
                        counts["new_energy_computed"] += 1
                    else:
                        counts["energy_source_gaps"] += 1
                        continue
                    energy_stream.write(json.dumps({"schema_version":VERSION,"episode_id":episode,"entity_id":entity_id,"task_id":entity["task_id"],"tick":tick,
                        "source_class":"declared_model_computation_not_battery_telemetry","origin":origin,"values":values,
                        "source_refs":[str(dataset.task_plan_path),str(dataset.frames_path),str(episode_dir/"weather_meta.jsonl"),str(charging_path),str(profile_path)],
                        "replaces_frozen_observation":False,"p01_imported":False})+"\n")
                emit_status({"episode_id":episode,"entity_id":entity_id,"field":"global_uav_lifetime","status":"wiring_missing" if "activation_tick" not in entity else "available",
                    "resolved_by":"global_uav_lifetimes.jsonl","world_presence":"task_plan_declared","world_observation_status":"observer_unseen" if not ticks else "world_ledger_observed",
                    "scope":"frozen_episode_global_uav","source_ref":str(dataset.task_plan_path)})
                emit_status({"episode_id":episode,"entity_id":entity_id,"field":"state_of_charge_ratio","status":"source_unrecorded" if gap else "available",
                    "source_gap":gap,"world_observation_status":"observer_unseen" if not ticks else "world_ledger_observed", "resolved_by":"global_uav_energy.jsonl" if histories else None,
                    "scope":"current_episode_model_history","telemetry_status":"source_unrecorded","existing_numerical_values_preserved":True})
            emit_status({"episode_id":episode,"field":"task_deadline_missed","status":"source_unrecorded",
                "scope":"frozen_compute_comm_aggregate_world","source_ref":str(project/"aw_data/objective_semantic_truth"/episode/"simulation_logs/compute_log.jsonl"),
                "reason":"aggregate_quality_model_has_no_task_arrival_service_completion_deadline_source",
                "candidate_source_index":"compute_source_index.jsonl","candidates_adopted":False})
            emit_status({"episode_id":episode,"field_family":"communication_latency_and_packet_loss","status":"available",
                "scope":"frozen_compute_comm_aggregate_quality_model","source_ref":str(project/"aw_data/objective_semantic_truth"/episode/"simulation_logs/communication_log.jsonl"),
                "source_class":"quality_derived_model_estimates","tx_rx_measured":False,"native_packet_history_status":"source_unrecorded"})
            counts["episodes"] += 1
            if counts["episodes"] % 20 == 0:
                print(json.dumps({"progress_episodes":counts["episodes"],"new_energy_computed":counts["new_energy_computed"]}),flush=True)
        for arm_path in sorted((project/"arms").glob("*/*/*/*/manifest.json")):
            arm = _json(arm_path)
            if episode_filter and arm["episode_id"] != episode_filter:
                continue
            scope = declared_arm_scope(arm)
            if "predicate_ids" not in scope:
                for family in ("battery","compute"):
                    emit_status({"episode_id":arm["episode_id"],"arm_id":arm["arm_id"],"group_key":str(arm_path.parent.parent.relative_to(project/"arms")),"field_family":family,
                        "status":"wiring_missing","scope":scope,"window":arm["window"],"source_ref":str(arm_path),
                        "actual_ledger_ref":str(arm_path.parent/"objective/predicate_truth_ticks.jsonl"),
                        "reason":"manifest_declares_entity_scope_but_no_predicate_scope; core_applicability_not_assumed",
                        "factual_labels_copied":False,"p01_imported":False})
                counts["arms"] += 1
                continue
            predicates = scope["predicate_ids"]
            for family, token in (("battery","battery"),("compute","compute")):
                declared = any(token in predicate or (family == "battery" and any(t in predicate for t in ("energy","range_insufficient","power_derat"))) for predicate in predicates)
                emit_status({"episode_id":arm["episode_id"],"arm_id":arm["arm_id"],"group_key":str(arm_path.parent.parent.relative_to(project/"arms")),"field_family":family,
                    "status":"computation_missing" if declared else "not_applicable","scope":scope,
                    "window":arm["window"],"source_ref":str(arm_path),"actual_ledger_ref":str(arm_path.parent/"objective/predicate_truth_ticks.jsonl"),
                    "out_of_scope_immutable_core_ledger_ref":str(project/"aw_data/objective_semantic_truth"/arm["episode_id"]/"simulation_logs"/(family+"_log.jsonl")),
                    "factual_labels_copied":False,"p01_imported":False,
                    "reason":"declared_scope_requires_own_window_core_history" if declared else "outside_this_arm_declared_predicate_scope"})
            counts["arms"] += 1
    compute_counts = _produce_compute_index(project,compute_sources,output,episode_filter)
    result = {"schema_version":VERSION,"counts":dict(counts),"status_counts":dict(status_counts),"compute_counts":compute_counts,
        "products":{"lifetimes":"global_uav_lifetimes.jsonl","energy":"global_uav_energy.jsonl","field_status":"core_field_status.jsonl","compute_sources":"compute_source_index.jsonl","candidate_task_ledger":"compute_tasks.jsonl.gz"},
        "consumer_contract":"select an explicit source_id and regime; candidates are separate worlds, not historical replacements",
        "p01_import_contract_changed":False,"frozen_episode_and_arm_bytes_changed":False,
        "current_p01_import":{"module":"Dataset/world_model/p01_schema_rollout_v4/src/p01v4/data/import_episode.py",
            "reads":["truth_frames.jsonl","global_entity_roster.json","world_truth_graph_base.json.initial_assertions","world_truth_graph_deltas.jsonl","event_occurrences.jsonl"],
            "new_core_side_tables_consumed":False,"new_core_labels_in_current_p01_head":False},
        "typed_truth_producer":{"module":"Dataset/semantic_simulation/domain_state.py","family":"payload_energy",
            "connector":"domain_global_energy","shared_computation":"world_energy_history -> energy_step",
            "requires_regeneration_before_P01_receives_repaired_typed_truth":True},
        "energy_source_kind":"declared_parameter_model_evaluated_on_real_simulator_motion_and_episode_weather",
        "native_physics_or_battery_telemetry_claimed":False,
        "limitations":["Pre-episode temperature, charging and initial energy history are unrecorded; current weather is never backfilled.",
            "ARM target truth remains arm-local; out-of-scope core ledgers are immutable source references.",
            "Compute candidate resource models do not supply missing historical TX/RX data or adopt business load as frozen world truth."]}
    index_path = output/"source_index.json"
    existing = _json(index_path) if index_path.is_file() else {}
    existing.update(result)
    _write_json(index_path,existing)
    return result


def _produce_compute_index(project: Path,roots: list[tuple[Path,str,str]],output: Path,episode_filter: str | None) -> dict[str,int]:
    counts = Counter()
    roster_ids: dict[str,set[str]] = {}
    with (output/"compute_source_index.jsonl").open("w") as stream,gzip.open(output/"compute_tasks.jsonl.gz","wt",encoding="utf-8") as tasks:
        for root,applicability,task_filename in roots:
            if not root.is_dir():
                raise FileNotFoundError(root)
            for config_path in sorted(root.glob("*/*/run_config.json")):
                config = _json(config_path)
                episode = config["episode_id"]
                if episode_filter and episode != episode_filter:
                    continue
                source_dir = config_path.parent
                task_path = source_dir/task_filename
                if not task_path.is_file():
                    raise FileNotFoundError(task_path)
                source = {"schema_version":VERSION,"source_id":str(source_dir),"episode_id":episode,
                    "regime":config["profile"],"applicability":applicability,"config_path":str(config_path),"tasks_path":str(task_path),
                    "owner_entity_ids":config["owner_entity_ids"],"field_provenance":config["field_provenance"],
                    "time_unit":"ns","clock":"episode_simulation_time","cycles_unit":"cycle","memory_unit":"byte",
                    "source_class":"authored_deterministic_resource_simulation","adopted_as_frozen_world":False,"p01_imported":False}
                if episode not in roster_ids:
                    roster_ids[episode] = {row["entity_id"] for row in _json(project/"aw_data/render_ready_episodes_capture_filtered"/episode/"global_entity_roster.json")["entities"]}
                source["owner_binding_status"] = {owner:"available" if owner in roster_ids[episode] else "conversion_error" for owner in source["owner_entity_ids"]}
                source["owner_binding_reference"] = str(project/"aw_data/render_ready_episodes_capture_filtered"/episode/"global_entity_roster.json")
                source["usable_for_current_entity_binding"] = all(owner in roster_ids[episode] for owner in source["owner_entity_ids"])
                summary = _json(source_dir/"summary.json")
                source["reported_task_count"] = summary["tasks"]
                stream.write(json.dumps(source)+"\n")
                counts["sources"] += 1
                counts[applicability] += 1
                # Large low-load references remain path-indexed immutable sources.
                if applicability == "LOW_LOAD_REFERENCE_ONLY":
                    continue
                for row in read_task_source(source):
                    tasks.write(json.dumps(row)+"\n")
                    counts["candidate_tasks_consumed"] += 1
                    if row["deadline_missed"] is True: counts["candidate_deadline_misses"] += 1
                    elif row["deadline_missed"] is False: counts["candidate_on_time_or_nonmiss_terminal"] += 1
                    else: counts["candidate_censored_outcomes"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output",type=Path)
    parser.add_argument("--compute-source",nargs=3,action="append",required=True,
        metavar=("DIRECTORY","APPLICABILITY","TASK_FILENAME"),
        help="Explicit retained task-source directory and scientific scope; repeat for separate regimes")
    parser.add_argument("--episode")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (args.output or project/"aw_data/domain_state_supplement").resolve()
    compute_sources = [(Path(directory).resolve(), applicability, filename)
                       for directory, applicability, filename in args.compute_source]
    print(json.dumps(produce(project,output,compute_sources=compute_sources,episode_filter=args.episode),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()

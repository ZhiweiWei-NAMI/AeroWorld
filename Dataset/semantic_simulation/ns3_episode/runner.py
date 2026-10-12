"""Sequential, resumable full-episode production in a new output namespace."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path

from . import PRODUCER_VERSION
from .compute import simulate_local_tasks
from .event_logs import make_artifact_files, resolve_artifact_ref, write_episode_logs, write_jsonl
from .provider_adapter import DEFAULT_IMAGE, ProviderAdapter, default_radio_config, resolve_radio_config
from .rule_binding import CONTRACT_VERSION, bind_predicates
from .trajectory import load_episode

ROOT = Path(__file__).resolve().parents[3]
PRODUCTS = (
    "run_config.json", "node_registry.jsonl", "network_events.jsonl",
    "network_packets.jsonl", "network_queue_samples.jsonl", "network_diagnostics.jsonl", "compute_events.jsonl", "compute_tasks.jsonl",
    "network_states.jsonl", "compute_states.jsonl", "predicates.jsonl", "summary.json",
)
LOAD_PROFILES = {
    "covered_load_0p5m": 500_000,
    "covered_load_2m": 2_000_000,
    "covered_load_4m": 4_000_000,
    "covered_load_6m": 6_000_000,
}
PROFILES = ("baseline", "controlled_congestion", "controlled_compute", "timed_intervention", *LOAD_PROFILES)


def make_config(episode, profile, image, log_compression="none", radio=None, radio_source_ref=None,
                geometry_config=None, geometry_source_ref=None, geometry_metadata=None,
                intervention=None, intervention_source_ref=None, gateway_selection_ref=None):
    """Declared assumptions, separate from all measured provider outputs."""
    arrival_period_ns = 25_000_000 if profile == "controlled_compute" else 1_000_000_000
    config = {
        "producer_version": PRODUCER_VERSION,
        "producer_contract_version": CONTRACT_VERSION,
        "output_format_version": "ns3_episode_v5_radio",
        "log_compression": log_compression, "artifact_files": make_artifact_files(log_compression),
        "episode_id": episode["episode_id"], "profile": profile, "seed": 1, "run": 1,
        "network_profile": "controlled_congestion" if profile == "controlled_congestion" else "baseline",
        "image": image, "duration_ns": episode["duration_ns"],
        "source_ref": episode["source_ref"], "state_period_ns": 500_000_000,
        "gateway_selection": {"episode_id":episode["episode_id"],
            "entity_id":episode["gateway_entity_id"], "source_ref":gateway_selection_ref},
        "state_semantics": "events strictly before observation time; horizon is censoring",
        "network": {"packet_ttl_ns": 200_000_000, "loss_window_ns": 1_000_000_000,
            "cohort_anchor": "packet_expiry_ns",
            "latency_population": "on-time delivered mature socket-accepted unique datagrams",
            "loss_population": "mature socket-accepted unique datagrams",
            "telemetry_payload_bytes": 256, "telemetry_period_ns": 100_000_000,
            "congestion_uav_owners": episode["roster_uav_ids"][:2] if profile == "controlled_congestion" else [],
            "congestion_bps_per_owner": 4_000_000, "congestion_payload_bytes": 1200},
        "radio": default_radio_config() if radio is None else radio,
        "radio_config_source_ref": "Dataset/semantic_simulation/ns3_episode/provider_adapter.py" if radio_source_ref is None else radio_source_ref,
        "radio_geometry": {"mode": "ad_hoc", "band_ghz": 2.4,
            "gateway_node_id": episode["gateway_node_id"],
            "gateway_antenna_offset_m": episode["gateway_antenna_offset_m"],
            "mobile_antenna_offset_m": [0, 0, 0], "shared_channel": True},
        "compute": {"profile_id": "p01.local.dedicated_fifo.v1", "cpu_hz": 2_000_000_000, "cycles_per_job": 100_000_000,
            "arrival_period_ns": arrival_period_ns, "deadline_ns": 100_000_000,
            "memory_bytes": 67_108_864,
            "theoretical_service_capacity_jobs_per_s": 20,
            "offered_jobs_per_s": 1_000_000_000 / arrival_period_ns,
            "offered_utilization": 50_000_000 / arrival_period_ns,
            "task_storage_policy": "one record per arrival; finite source availability and episode horizon bound storage",
            "memory_model": "declared per-job memory footprint; no memory admission or queue storage capacity model",
            "full_horizon_max_jobs_per_epoch": (episode["duration_ns"] + arrival_period_ns - 1) // arrival_period_ns,
            "model": "synthetic_dedicated_core_local_nonpreemptive_fifo"},
        "rules": {"L": {"field": "latency_mean_ms", "operator": "gt", "threshold": 120.0},
            "P": {"field": "packet_loss_ratio", "operator": "gt", "threshold": 0.05}},
        "unintegrated_operands": "legacy compute, offload, cellular and legacy MAC-rule operands; actual MAC queue snapshots and bounded radio diagnostics are observed separately",
        "schema_refs": ["Dataset/semantic_simulation/ns3_episode/schemas/" + name for name in
            ("node_registry.schema.json", "network_event.schema.json", "network_packet.schema.json", "network_state.schema.json",
             "network_queue_sample.schema.json", "network_diagnostics.schema.json", "compute_event.schema.json", "compute_task.schema.json", "compute_state.schema.json")],
    }
    if profile in LOAD_PROFILES:
        config["network_profile"] = "coverage_load"
        config["network"].update({
            "traffic_owner_ids": ["global_uav_uav_intersection_inspect_06"],
            "payload_bytes": 1200, "load_bps": LOAD_PROFILES[profile],
            "period_ns": 1200 * 8 * 1_000_000_000 // LOAD_PROFILES[profile],
            "telemetry_enabled": False, "other_apps_enabled": False,
            "selection_basis": "Explicit approved owner selected from first v1 connected coverage and geometry; no future labels or performance-based reselection.",
            "selection_source_ref": "aw_data/ns3_episode_v1/L6-2_v1__seed00/baseline/run1/network_packets.jsonl",
        })
    if geometry_config is not None:
        config["output_format_version"] = "ns3_episode_v6_geometry"
        config["artifact_files"]["network_geometry_samples.jsonl"] = "network_geometry_samples.jsonl" + (".gz" if log_compression == "gzip" else "")
        config["schema_refs"].append("Dataset/semantic_simulation/ns3_episode/schemas/network_geometry_sample.schema.json")
        config["geometry"] = {
            "source_ref": geometry_source_ref, "configuration": geometry_config,
            "export_metadata": geometry_metadata,
            "applied_support": "R2_geometric_LOS_NLOS_close_in_only",
            "applied_shadowing": False, "extra_fast_fading": False,
            "shadowing": {"applied": False, "status": "pending_supported_correlated_symmetric_link_implementation",
                "sigma_los_db": geometry_config["propagation_prior"]["sigma_los_db"],
                "sigma_nlos_db": geometry_config["propagation_prior"]["sigma_nlos_db"],
                "correlation_distance_m": geometry_config["propagation_prior"]["shadowing_correlation_distance_m"]},
        }
    if profile == "timed_intervention":
        if intervention is None:
            raise ValueError("timed_intervention requires an explicit intervention JSON")
        if set(intervention) != {"schema_version","episode_id","traffic_windows","source_ref","mechanism","assumptions"}:
            raise ValueError("intervention JSON requires exactly the canonical top-level fields")
        if intervention["schema_version"] != "p01.ns3.timed-intervention/v1" or intervention["episode_id"] != episode["episode_id"]:
            raise ValueError("intervention schema/episode differs from the declared run")
        if not isinstance(intervention["traffic_windows"],list) or not intervention["traffic_windows"]:
            raise ValueError("timed intervention requires explicit nonempty traffic windows")
        config["output_format_version"] = "ns3_episode_v7_timed_flow"
        config["network_profile"] = "timed_flow"
        config["network"]["traffic_windows"] = intervention["traffic_windows"]
        config["network"]["telemetry_enabled"] = False
        config["network"]["other_apps_enabled"] = False
        config["intervention"] = {"source_ref":intervention_source_ref,"configuration":intervention}
    return config


def registry_records(episode, config):
    """Map source-present radio epochs to the supplied formal registry schema.

    The replay's internal interval end is observed availability departure. It
    provides no physical death evidence, including when departure precedes H.
    """
    records = []
    for node in episode["nodes"]:
        infrastructure = node["node_type"] == "gateway"
        source = node["source_ref"]
        offset = episode["gateway_antenna_offset_m"] if infrastructure else [0, 0, 0]
        position = [coordinate - antenna for coordinate, antenna in zip(node["position_enu_m"], offset)]
        observation_end = episode["duration_ns"] if node["death_ns"] is None else node["death_ns"]
        def metadata(status, assumption=None, calibration_need=None):
            return {"status": status, "source": [source, "run_config.json"],
                    "assumption": assumption, "calibration_need": calibration_need}
        records.append({
            "episode_id": episode["episode_id"], "node_id": node["node_id"],
            "ns3_node_id": node["ns3_node_id"],
            "owner_entity_id": None if infrastructure else node["owner"],
            "infra_id": node["infra_id"] if infrastructure else None,
            "life_epoch": node["life_epoch"],
            "roles": ["authored_site", "wifi_gateway"] if infrastructure else
                     [{"uav": "uav_terminal", "vehicle": "vehicle_terminal"}[node["node_type"]]],
            "position_authority": {
                "kind": "authored_station" if infrastructure else "truth_actor",
                "source": source, "field": "entities[].truth_pose.position_enu_m",
                "site_position_enu_m": position if infrastructure else None},
            "coordinate_contract": "coord.external_enu_m.v1",
            "antenna_offset_enu_m": offset,
            "active_intervals_ns": [{"birth_ns": node["birth_ns"], "death_ns": None,
                "observation_end_ns": observation_end, "right_censored": True}],
            "radio_profile_ids": [config["radio"]["profile_id"]],
            "compute_profile_id": config["compute"]["profile_id"] if node["node_type"] == "uav" else None,
            "field_metadata": {
                "owner_entity_id": metadata("observed_current"),
                "infra_id": metadata("observed_current"),
                "ns3_node_id": metadata("derived_from_current", "Run-local index in the explicit node registry order."),
                "life_epoch": metadata("derived_from_current", "Source-present radio availability epoch; no physical birth/death inference."),
                "roles": metadata("proposal_assumption", "UAV/vehicle radio terminal; authored station acts as Wi-Fi gateway."),
                "position_authority": metadata("observed_current"),
                "coordinate_contract": metadata("observed_current"),
                "antenna_offset_enu_m": metadata("proposal_assumption", "Authored station +[0,0,3]m; mobile terminals zero offset.", "Radio antenna geometry calibration."),
                "active_intervals_ns": metadata("derived_from_current", "birth_ns is first source presence; observation_end_ns is observed radio availability departure or horizon. Physical death is unobserved; death_ns remains null and right_censored true."),
                "radio_profile_ids": metadata("proposal_assumption", "Declared synthetic shared-channel Wi-Fi profile.", "Propagation and radio calibration."),
                "compute_profile_id": metadata("proposal_assumption", "Synthetic dedicated single-core FIFO profile for UAV terminals.", "CPU and job workload calibration."),
            },
        })
    return records


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def completed_run(output_dir, config):
    """A final summary marks completion; incomplete outputs remain visible."""
    output = Path(output_dir)
    if not output.exists() or not any(output.iterdir()):
        return None
    products = ("run_config.json", "summary.json", *config["artifact_files"].values())
    if not all((output / name).is_file() for name in products):
        raise FileExistsError(f"Incomplete run output; preserve and inspect it: {output}")
    previous = json.loads((output / "run_config.json").read_text())
    if previous != config:
        raise ValueError(f"Completed run configuration differs: {output}")
    return json.loads((output / "summary.json").read_text())


def run_episode(episode, config, output_dir, provider, geometry_export=None):
    """Run the provider directly, then project its actual events once."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files, compression = config["artifact_files"], config["log_compression"]
    _json(output / "run_config.json", config)
    write_jsonl(output / files["node_registry.jsonl"], registry_records(episode, config), compression=compression)
    network = provider.run_episode(episode, config, output, geometry_export=geometry_export)
    compute = simulate_local_tasks(episode, config)
    projected = write_episode_logs(output, episode, config, network, compute)
    predicates = bind_predicates(episode, projected["network_states"], projected["compute_states"], config)
    for row in predicates:
        row["source_ref"] = resolve_artifact_ref(row["source_ref"], files)
    write_jsonl(output / files["predicates.jsonl"], predicates, compression=compression)
    summary = projected["summary"]
    summary.update({"profile": config["profile"], "seed": config["seed"], "run": config["run"],
        "producer_contract_version": CONTRACT_VERSION,
        "output_format_version": config["output_format_version"], "log_compression": compression,
        "artifact_files": files,
        "node_count": len(episode["nodes"]), "mobility_time_count": len(episode["mobility"]),
        "predicate_rows": len(predicates),
        "predicate_truth_counts": {symbol: dict(Counter(row["value"] for row in predicates if row["symbol"] == symbol))
                                   for symbol in ("L", "P")}})
    summary["output_bytes"].update({name: (output / name).stat().st_size for name in
        ("run_config.json", files["node_registry.jsonl"], files["predicates.jsonl"])})
    _json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", required=True, help="Explicit frozen episode IDs")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES)
    parser.add_argument("--output-root", type=Path, default=Path("aw_data/ns3_episode_v5_radio"))
    parser.add_argument("--log-compression", choices=("none", "gzip"), default="none")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--radio-json", type=Path, help="Complete radio parameter JSON with exact units and provenance")
    parser.add_argument("--geometry-json", type=Path, help="Explicit R2 geometry configuration; absent disables building obstruction")
    parser.add_argument("--gateway-map-json", type=Path, required=True,
                        help="Exact JSON episode-ID to authored gateway entity-ID mapping")
    parser.add_argument("--intervention-json", type=Path,
                        help="Canonical p01.ns3.timed-intervention/v1 document")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    gateway_source = args.gateway_map_json if args.gateway_map_json.is_absolute() else args.root / args.gateway_map_json
    gateway_entities = json.loads(gateway_source.read_text())
    intervention_source = args.intervention_json
    if intervention_source is not None and not intervention_source.is_absolute():
        intervention_source = args.root / intervention_source
    intervention = None if intervention_source is None else json.loads(intervention_source.read_text())
    profiles = args.profiles if args.profiles is not None else (["timed_intervention"] if intervention is not None else ["baseline","controlled_congestion"])
    if intervention is not None and "timed_intervention" not in profiles:
        parser.error("--intervention-json requires the timed_intervention profile")
    radio_source = args.radio_json
    if radio_source is not None and not radio_source.is_absolute():
        radio_source = args.root / radio_source
    radio = resolve_radio_config(default_radio_config() if radio_source is None else json.loads(radio_source.read_text()))
    geometry_config, geometry_export, geometry_source = None, None, args.geometry_json
    if geometry_source is not None:
        from .geometry import load_building_geometry
        if not geometry_source.is_absolute():
            geometry_source = args.root / geometry_source
        geometry_config = json.loads(geometry_source.read_text())
        geometry_export = load_building_geometry(args.root, config_path=geometry_source).export_for_ns3()
    output_root = args.output_root if args.output_root.is_absolute() else args.root / args.output_root
    with ExitStack() as stack:
        provider = None
        for episode_id in dict.fromkeys(args.episodes):
            episode = load_episode(args.root, episode_id, gateway_entity_id=gateway_entities[episode_id])
            for profile in dict.fromkeys(profiles):
                config = make_config(episode, profile, args.image, args.log_compression, radio,
                                     None if radio_source is None else str(radio_source), geometry_config,
                                     None if geometry_source is None else str(geometry_source),
                                     None if geometry_export is None else geometry_export["metadata"],
                                     intervention, None if intervention_source is None else str(intervention_source),
                                     str(gateway_source))
                output = output_root / episode_id / profile / "run1"
                summary = completed_run(output, config)
                resumed = summary is not None
                if not resumed:
                    if provider is None:
                        provider = stack.enter_context(ProviderAdapter(image=args.image))
                    summary = run_episode(episode, config, output, provider, geometry_export)
                print(json.dumps({"episode_id": episode_id, "profile": profile, "output_dir": str(output),
                    "resumed": resumed, "state_time_count": summary["state_time_count"],
                    "network": {key: summary["network"][key] for key in
                        ("packet_count", "mature_at_horizon_count", "mature_ttl_loss_count", "late_rx_count", "wall_time_s", "peak_rss_kb")},
                    "compute": {key: summary["compute"][key] for key in
                        ("task_count", "task_deadline_missed_count", "task_deadline_unknown_count")}}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

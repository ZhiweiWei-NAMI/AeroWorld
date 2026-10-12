"""Explicit R1 mature-datagram L/P inputs for the generic trajectory interface.

Observed future labels are separate from prediction inputs. Replay is an
assembly exercise, never a prediction or a training-ready batch. Scenario-level
splits bind every seed/profile/run of the same declared scene to one cohort.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from Dataset.world_model.rule_factor_graph.generic_grounding import ParameterProfile, RoleBinding, TypedEntity
from Dataset.world_model.rule_factor_graph.unified_contracts import ArgumentRole, FieldSchema, PredicateContract, ContractError
from Dataset.world_model.rule_factor_graph.predicate_trajectory_data import (
    EpisodeContract, CandidateTrajectory, PredicateQuery, FuturePredicateEvaluator,
    TRUTH_CODES, STATE_OFFSETS, formula_fields, assemble_trajectory_batch,
    to_trajectory_pyg,
)

PRODUCER_CONTRACT = "p01.ns3.mature-datagram-lp/v1"
ADAPTER_VERSION = "p01.ns3.training-adapter/v1"
TICK_NS = 100_000_000
GRID_NS = 500_000_000
# Source predicate IDs identify admitted log comparisons; these are NOT legacy
# producer bindings. The newly constructed formulas have a separate authority.
RULES = (
    ("L", "communication.data_flow_latency_above_threshold", "latency_mean_ms", "ms", "time", "latency_threshold_ms", 120.0),
    ("P", "communication.packet_loss_above_threshold", "packet_loss_ratio", "1", "dimensionless", "loss_threshold", .05),
)


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


@dataclass(frozen=True)
class RunSelection:
    episode_id: str
    profile: str
    run: int


@dataclass(frozen=True)
class FlowKey:
    episode_id: str
    profile: str
    run: int
    link_id: str
    flow_id: str
    src_node_id: str
    owner: str
    life_epoch: int
    dst_node_id: str
    receiver_owner: str
    receiver_life_epoch: int


@dataclass(frozen=True)
class WindowKey:
    flow: FlowKey
    anchor_ns: int


@dataclass(frozen=True)
class PredictionPrefix:
    key: WindowKey
    episode: EpisodeContract
    history: np.ndarray
    history_known: np.ndarray
    history_applicable: np.ndarray
    source_ref: str


@dataclass(frozen=True)
class WindowSupervision:
    targets: np.ndarray
    known: np.ndarray
    compatible: np.ndarray
    observed_truth_codes: np.ndarray
    source_ref: str


@dataclass(frozen=True)
class CandidateOperands:
    values: np.ndarray
    known: np.ndarray
    applicable: np.ndarray
    source_ref: str
    provenance: str


@dataclass(frozen=True)
class CandidateLawDeclaration:
    """Declared once; fitting authority is TRAIN physical episodes only."""
    method: str
    fit_episode_ids: tuple[str, ...]
    source_ref: str


class ScopedFormulaEvaluator(FuturePredicateEvaluator):
    """Keep supplied candidate applicability distinct from missing operands."""
    def __init__(self, episode, applicable):
        super().__init__(episode)
        self.applicable = _readonly(applicable, bool)

    def evaluate(self, record):
        codes = super().evaluate(record)
        if self.applicable.shape != codes.shape:
            raise ContractError("candidate applicability differs from fixed-rule result axes")
        return _readonly(np.where(self.applicable, codes, TRUTH_CODES["out_of_scope"]), np.int8)


def _contracts(source_ref):
    roles = (ArgumentRole("source_owner", "world:UnmannedAircraft"),
             ArgumentRole("receiver_owner", "world:CommunicationStation"),
             ArgumentRole("flow", "world:DataFlow"))
    result = []
    for symbol, _, field, unit, _, parameter, _ in RULES:
        path = f"p01_ns3.{field}"
        result.append(PredicateContract(
            predicate_id=f"p01.ns3.mature_datagram_lp.v1.{symbol}", roles=roles,
            required_fields=(FieldSchema(path, "numeric", unit, f"{PRODUCER_CONTRACT}:{field}"),),
            threshold_parameters=(parameter,), evaluation_spec={"op": "gt", "field": path, "threshold": parameter},
            grounding_spec={"candidate_authority": {"source": PRODUCER_CONTRACT},
                "field_sources": {path: f"network_states:{field}.value"},
                "field_owner_roles": {path: ("source_owner", "receiver_owner", "flow")},
                "distinct_role_sets": (("source_owner", "receiver_owner", "flow"),),
                "unordered_role_sets": (), "role_sources": {}},
            operators=("gt",), executable=True, semantic_mode="declared_formula",
            physical_formula=f"{field} > governed {parameter}",
            semantic_ambiguity="Mature datagram producer semantics; no legacy generator relation admitted.",
            source_refs=(source_ref, ADAPTER_VERSION), template={"producer_contract_version": PRODUCER_CONTRACT}))
    return tuple(result)


def _key(row, selection):
    return FlowKey(selection.episode_id, selection.profile, selection.run,
        row["link_id"], row["flow_id"], row["src_node_id"], row["owner"], row["life_epoch"],
        row["dst_node_id"], row["receiver_owner"], row["receiver_life_epoch"])


def _metric(row, field, unit, dimension):
    metric = row[field]
    if (metric["unit"], metric["dimension"]) != (unit, dimension):
        raise ContractError("new producer field unit/dimension differs from admitted contract")
    if metric["status"] == "KNOWN":
        value = metric["value"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ContractError("KNOWN producer operand must be finite numeric")
        return float(value), True
    if metric["status"] != "UNKNOWN" or metric["value"] is not None or not metric["reason"]:
        raise ContractError("missing producer operand needs explicit UNKNOWN and reason")
    return np.nan, False


class R1TrainingLogs:
    def __init__(self, *, split_by_episode, scenario_by_episode, states, predicates, flows, configs):
        self.split_by_episode = MappingProxyType(dict(split_by_episode))
        self.scenario_by_episode = MappingProxyType(dict(scenario_by_episode))
        self._states, self._predicates, self._flows, self.configs = states, predicates, flows, configs
        self._declared_laws = set()

    def eligible_windows(self, *, split, history_steps=8):
        """Full-grid prefixes, horizon-bounded futures, at least one known anchor."""
        if split not in ("train", "validation") or type(history_steps) is not int or history_steps < 1:
            raise ValueError("window construction needs an explicit split and history length")
        windows = []
        for key, authority in self._flows.items():
            if self.split_by_episode[key.episode_id] != split:
                continue
            for t in range((history_steps - 1)*GRID_NS, authority["duration_ns"] - 50*TICK_NS + 1, GRID_NS):
                row = self._states[(key, t)]
                if row["status"] != "IN_SCOPE":
                    continue
                if not any(self._predicates[(key, t, symbol)]["value"] in ("true", "false") for symbol, *_ in RULES):
                    continue
                windows.append(WindowKey(key, t))
        return tuple(windows)

    def prediction_prefix(self, window, *, history_steps=8):
        """Return only observed prefix values/metadata; no future or labels."""
        authority = self._flows[window.flow]
        times = tuple(window.anchor_ns - (history_steps - 1 - i)*GRID_NS for i in range(history_steps))
        values, known, applicable = [], [], []
        for t in times:
            row = self._states[(window.flow, t)]
            scope = row["status"] == "IN_SCOPE"
            numeric = [_metric(row, field, unit, dim) for _, _, field, unit, dim, _, _ in RULES]
            values.append([v if scope else np.nan for v, _ in numeric])
            known.append([observed and scope for _, observed in numeric])
            applicable.append([scope]*len(RULES))
        episode = EpisodeContract(window.flow.episode_id,
            f"{window.flow.profile}/run{window.flow.run}/{window.flow.flow_id}",
            self.split_by_episode[window.flow.episode_id], tuple(t//TICK_NS for t in times),
            window.anchor_ns//TICK_NS, authority["fields"], authority["queries"])
        return PredictionPrefix(window, episode, _readonly(values), _readonly(known, bool),
                                _readonly(applicable, bool), authority["state_ref"])

    def supervision(self, window):
        """Observed cumulative first-opposite evidence, independent of candidates.

        After an observed opposite, F(t)=1 remains known even if later samples
        are UNKNOWN/OOS. Before any opposite, F(t)=0 requires every earlier
        observation to be known and equal to the known anchor.
        """
        codes = np.array([[TRUTH_CODES[self._predicates[(window.flow, window.anchor_ns+offset*TICK_NS, symbol)]["value"]]
                          for symbol, *_ in RULES] for offset in STATE_OFFSETS], dtype=np.int8)
        targets = np.full((2, 10), np.nan)
        known = np.zeros((2, 10), bool)
        compatible = np.ones((2, 11), bool)
        for p in range(2):
            anchor = codes[0, p]
            if anchor not in (TRUTH_CODES["false"], TRUTH_CODES["true"]):
                continue
            opposite_seen, complete = False, True
            for j, value in enumerate(codes[1:, p]):
                observed = value in (TRUTH_CODES["false"], TRUTH_CODES["true"])
                complete = complete and observed
                opposite_seen = opposite_seen or (observed and value != anchor)
                if opposite_seen or complete:
                    known[p, j] = True
                    targets[p, j] = float(opposite_seen)
                    if opposite_seen:
                        compatible[p, j+1:] = False
                    else:
                        compatible[p, :j+1] = False
        return WindowSupervision(_readonly(targets), _readonly(known), _readonly(compatible),
            _readonly(codes), self._flows[window.flow]["predicate_ref"])

    def declare_candidate_law(self, declaration):
        if not declaration.method or not declaration.source_ref or not declaration.fit_episode_ids:
            raise ValueError("candidate law needs explicit method, source and TRAIN fitting episodes")
        if any(self.split_by_episode[episode] != "train" for episode in declaration.fit_episode_ids):
            raise ValueError("candidate law fitting may use TRAIN physical episodes only")
        self._declared_laws.add(declaration)
        return declaration

    def assemble_prediction(self, prefix, candidates, *, law):
        """Assemble external predictions only; prefix object exposes no future truth."""
        if law not in self._declared_laws or candidates.provenance != "predicted_future_operands":
            raise ValueError("prediction needs a declared law and predicted future operands")
        return _assemble(prefix, candidates)

    def prepare_train_window(self, prefix, candidates, *, law, device, field_type_keys=None,
                             prior_logits=None):
        """Prepare generic TRAIN tensors; no optimizer or candidate fitting runs.

        External candidate operands enter assembly; observed cumulative labels
        are attached separately afterward. No replay batch can enter this path.
        A prior is optional and must be supplied by its declared external adapter.
        """
        if prefix.episode.split != "train":
            raise ValueError("TRAIN preparation requires a TRAIN physical episode")
        assembled = self.assemble_prediction(prefix, candidates, law=law)
        labels = self.supervision(prefix.key)
        from Dataset.world_model.rule_factor_graph.predicate_trajectory_training import prepare_assembled_batch
        return prepare_assembled_batch(assembled["batch"], split="train", device=device,
            targets=labels.targets[None], target_known=labels.known[None],
            compatible=labels.compatible[None], prior_logits=prior_logits,
            field_type_keys=field_type_keys)

    def exercise_observed_replay(self, prefix):
        """Read observed future OPERANDS for assembly checks, never training input."""
        rows = [self._states[(prefix.key.flow, prefix.key.anchor_ns+offset*TICK_NS)] for offset in STATE_OFFSETS]
        values, known, scope = [], [], []
        for row in rows:
            active = row["status"] == "IN_SCOPE"
            numeric = [_metric(row, field, unit, dim) for _, _, field, unit, dim, _, _ in RULES]
            values.append([value if active else np.nan for value, _ in numeric])
            known.append([observed and active for _, observed in numeric])
            scope.append([active]*2)
        candidates = CandidateOperands(_readonly([values]), _readonly([known], bool),
            _readonly([scope], bool), prefix.source_ref, "observed_replay_assembly_only")
        return _assemble(prefix, candidates)

    def label_counts(self, windows):
        counts = {"windows": len(windows), "known_anchor_predicates": 0, "known_cdf_cells": 0,
                  "unknown_cdf_cells": 0, "known_positive_cdf_cells": 0,
                  "observed_unknown_truth_cells": 0, "observed_out_of_scope_truth_cells": 0}
        for window in windows:
            labels = self.supervision(window)
            counts["known_anchor_predicates"] += int(np.isin(labels.observed_truth_codes[0], (0,1)).sum())
            counts["known_cdf_cells"] += int(labels.known.sum())
            counts["unknown_cdf_cells"] += int((~labels.known).sum())
            counts["known_positive_cdf_cells"] += int(np.count_nonzero(labels.targets[labels.known]))
            counts["observed_unknown_truth_cells"] += int((labels.observed_truth_codes == TRUTH_CODES["unknown"]).sum())
            counts["observed_out_of_scope_truth_cells"] += int((labels.observed_truth_codes == TRUTH_CODES["out_of_scope"]).sum())
        return counts


def _assemble(prefix, candidates):
    if not candidates.source_ref or candidates.provenance not in ("predicted_future_operands", "observed_replay_assembly_only"):
        raise ValueError("candidate operands require explicit source and supported provenance")
    if not np.array_equal(candidates.applicable[:, 0, :],
                          np.broadcast_to(prefix.history_applicable[-1], candidates.applicable[:, 0, :].shape)):
        raise ContractError("candidate anchor availability differs from observed prefix")
    record = CandidateTrajectory(prefix.episode, prefix.history, prefix.history_known,
                                 candidates.values, candidates.known)
    evaluator = ScopedFormulaEvaluator(prefix.episode, candidates.applicable)
    batch = assemble_trajectory_batch((record,), evaluators=(evaluator,))
    pyg = to_trajectory_pyg(record, evaluator=evaluator)
    import torch
    pyg.graph["ObservedPrefixState"].applicable = torch.tensor(prefix.history_applicable.reshape(-1), dtype=torch.bool)
    return {"record": record, "batch": batch, "pyg": pyg, "candidate_provenance": candidates.provenance,
            "candidate_source_ref": candidates.source_ref,
            "training_ready": candidates.provenance == "predicted_future_operands"}


def load_r1_logs(root, selections, *, scenario_by_episode, split_by_scenario,
                 r1_root="aw_data/ns3_episode_v5_radio_reference/R1"):
    """Load explicitly selected runs; prepare exact joins once.

    scenario_by_episode is an explicit scene identity authority, not a filename
    heuristic. split_by_scenario prevents scene seeds/profiles/runs straddling
    TRAIN and VALID. Missing declarations or source keys fail explicitly.
    """
    root = Path(root)
    selections = tuple(selections)
    split_by_episode = {s.episode_id: split_by_scenario[scenario_by_episode[s.episode_id]] for s in selections}
    if any(not scenario_by_episode[s.episode_id] for s in selections):
        raise ValueError("physical scenario identity must be explicitly declared")
    states, predicates, flows, configs = {}, {}, {}, {}
    for selection in selections:
        split = split_by_episode[selection.episode_id]
        if split not in ("train", "validation"):
            raise ValueError("selected physical episode must have explicit TRAIN/VALID assignment")
        path = root/r1_root/selection.episode_id/selection.profile/f"run{selection.run}"
        config = json.loads((path/"run_config.json").read_text())
        if (config["episode_id"], config["profile"], config["run"], config["producer_contract_version"]) != (
                selection.episode_id, selection.profile, selection.run, PRODUCER_CONTRACT):
            raise ContractError("run identity or admitted producer contract differs")
        for symbol, _, field, _, _, _, threshold in RULES:
            if config["rules"][symbol] != {"field":field,"operator":"gt","threshold":threshold}:
                raise ContractError("governed R1 rule differs from admitted strict L/P comparison")
        config_key = (selection.episode_id, selection.profile, selection.run)
        configs[config_key] = config
        registry = {row["node_id"]: row for row in _jsonl(path/"node_registry.jsonl.gz")}
        state_ref, predicate_ref = str(path/"network_states.jsonl.gz"), str(path/"predicates.jsonl.gz")
        contracts = _contracts(state_ref)
        profile = ParameterProfile(f"{ADAPTER_VERSION}:{selection.profile}:run{selection.run}",
            {parameter:threshold for _,_,_,_,_,parameter,threshold in RULES},str(path/"run_config.json"))
        for row in _jsonl(path/"network_states.jsonl.gz"):
            if (row["episode_id"],row["source_ref"],row["config_ref"]) != (selection.episode_id,"network_packets.jsonl.gz","run_config.json") or row["status"] not in ("IN_SCOPE","OUT_OF_SCOPE"):
                raise ContractError("state episode or availability scope is not admitted")
            key = _key(row, selection)
            if key not in flows:
                src, dst = registry[key.src_node_id], registry[key.dst_node_id]
                if (src["owner_entity_id"],src["life_epoch"],dst["infra_id"],dst["life_epoch"]) != (key.owner,key.life_epoch,key.receiver_owner,key.receiver_life_epoch):
                    raise ContractError("state owners/epochs differ from exact node registry")
                if "uav_terminal" not in src["roles"] or "wifi_gateway" not in dst["roles"]:
                    raise ContractError("R1 field authority admits UAV-to-gateway roles only")
                bindings=(RoleBinding("source_owner",TypedEntity(key.src_node_id,"world:UnmannedAircraft")),
                    RoleBinding("receiver_owner",TypedEntity(key.dst_node_id,"world:CommunicationStation")),
                    RoleBinding("flow",TypedEntity(key.flow_id,"world:DataFlow")))
                queries=tuple(PredicateQuery(c,bindings,profile) for c in contracts)
                flows[key]={"queries":queries,"fields":tuple(f for q in queries for f in formula_fields(q)),
                    "state_ref":state_ref,"predicate_ref":predicate_ref,"duration_ns":config["duration_ns"]}
            if row["time_ns"] % GRID_NS or row["time_ns"] < 0:
                raise ContractError("network states must use the declared 2Hz integer clock")
            for _,_,field,unit,dim,_,_ in RULES:_metric(row,field,unit,dim)
            index=(key,row["time_ns"])
            if index in states:raise ContractError("duplicate exact state index")
            states[index]=row
        for row in _jsonl(path/"predicates.jsonl.gz"):
            if (row["episode_id"],row["profile"],row["producer_contract_version"],row["source_ref"]) != (
                    selection.episode_id,selection.profile,PRODUCER_CONTRACT,"network_states.jsonl.gz"):
                raise ContractError("predicate source/version/episode/profile differs")
            matches=[spec for spec in RULES if spec[0]==row["symbol"]]
            if len(matches)!=1:raise ContractError("predicate symbol has no admitted new-log rule")
            symbol,pid,field,unit,dim,_,threshold=matches[0]
            if (row["predicate_id"],row["operator"],row["threshold"],row["operand"]["field"],row["unit"],row["dimension"]) != (pid,"gt",threshold,field,unit,dim):
                raise ContractError("predicate comparison/operand authority differs")
            key=_key(row,selection);state=states[(key,row["time_ns"])]
            if row["scope_status"]!=state["status"] or {k:row['operand'][k] for k in ('value','status','reason','unit','dimension')} != state[field]:
                raise ContractError("predicate operands do not exactly join the observed source state")
            expected="out_of_scope" if state["status"]=="OUT_OF_SCOPE" else "unknown" if state[field]["status"]=="UNKNOWN" else "true" if state[field]["value"]>threshold else "false"
            if row["value"]!=expected:raise ContractError("logged truth differs from fixed admitted comparison")
            index=(key,row["time_ns"],symbol)
            if index in predicates:raise ContractError("duplicate exact predicate index")
            predicates[index]=row
    return R1TrainingLogs(split_by_episode=split_by_episode,scenario_by_episode=scenario_by_episode,states=states,predicates=predicates,flows=flows,configs=configs)

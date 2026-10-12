"""Typed communication graph, source-checked finite states and fixed rule support.

A is reduced record-level availability, Q is a threshold regime. Future A/Q
are latent. This finite domain models their joint mass; it does not integrate
continuous communication dynamics or identify an unobserved physical mode.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import yaml
from rdflib import Graph, Literal, Namespace

from .contracts import GroundingError, RoleBinding, StateKey, Variable, VariableKind

PREDICATE_ORDER = (
    "communication.data_flow_latency_above_threshold",
    "communication.packet_loss_above_threshold",
)
VARIABLE_NAMES = ("A", "Q", "L", "P")
STATE_FIELDS = ("station_operational_state.reduced_availability", "link_quality.threshold_regime")
DEFAULT_MIXED_ASSIGNMENTS = torch.tensor([
    [0, 0, 0, 0], [0, 1, 0, 1], [0, 2, 1, 1],
    [1, 0, 0, 0], [1, 1, 0, 1], [1, 2, 1, 1],
], dtype=torch.long)
PREDICATE_ASSIGNMENTS = DEFAULT_MIXED_ASSIGNMENTS[:, 2:].clone()
_ROOT = Path(__file__).resolve().parents[3]
_PROFILE = "Dataset/semantic_rules/profiles/compute_comm_supplement_profile.json"
_CONTRACT = "Dataset/semantic_rules/predicates/world_truth_predicate_contracts.yaml"
_ONTOLOGY = "Dataset/knowledge_graph/domain/communication_process.ttl"
_PRODUCER = "Dataset/semantic_simulation/compute_comm.py"
_AW = Namespace("https://w3id.org/aeroworld/ontology/core#")
_WORLD = Namespace("https://w3id.org/aeroworld/ontology/world#")
_DOM = Namespace("https://w3id.org/aeroworld/ontology/domain/communication_process#")


def joint_assignments(*, device=None) -> torch.Tensor:
    """Six legal rows, with columns A,Q,L,P and order A then Q."""
    return DEFAULT_MIXED_ASSIGNMENTS.to(device=device).clone()


def hard_threshold_support(assignments: torch.Tensor) -> torch.Tensor:
    """Fixed Q->L/P rule, including exact zero support for illegal assignments."""
    if not isinstance(assignments, torch.Tensor) or assignments.ndim != 2 or assignments.shape[1] != 4:
        raise ValueError("mixed assignments must have shape [S,4] in A,Q,L,P order")
    if assignments.is_floating_point() or assignments.is_complex():
        raise TypeError("mixed assignments must have an integer or Boolean dtype")
    a, q, l, p = assignments.unbind(-1)
    domain = ((a == 0) | (a == 1)) & (q >= 0) & (q <= 2)
    return domain & (l == (q == 2)) & (p == (q >= 1))


def compile_mixed_support(
    initial_joint_state: int | None = None, time_steps: int = 11, *,
    state_evidence: torch.Tensor | None = None, state_known: torch.Tensor | None = None,
    predicate_evidence: torch.Tensor | None = None, predicate_known: torch.Tensor | None = None,
    device=None,
) -> torch.Tensor:
    """Compile [T,6] support; explicit missing evidence never becomes false.

    State evidence columns are A,Q. Predicate evidence columns are L,P. This
    conditioning helper is separate from the neural prediction interface.
    Prediction callers provide only prefix evidence (normally the anchor).
    Training may intersect labels in a separate numerator partition.
    """
    if type(time_steps) is not int or not 1 <= time_steps <= 11:
        raise ValueError("mixed chain time budget is 1..11")
    if initial_joint_state is not None and (type(initial_joint_state) is not int or not 0 <= initial_joint_state < 6):
        raise ValueError("initial_joint_state must be a six-state index or None")
    states = joint_assignments(device=device)
    support = hard_threshold_support(states).expand(time_steps, -1).clone()
    if initial_joint_state is not None:
        support[0] = torch.arange(6, device=states.device) == initial_joint_state
    for values, known, columns, domains, name in (
        (state_evidence, state_known, (0, 1), (2, 3), "state"),
        (predicate_evidence, predicate_known, (2, 3), (2, 2), "predicate"),
    ):
        if values is None and known is None:
            continue
        if not isinstance(values, torch.Tensor) or not isinstance(known, torch.Tensor):
            raise TypeError(f"{name} evidence requires both values and known-mask tensors")
        if values.shape != (time_steps, 2) or known.shape != values.shape or known.dtype != torch.bool:
            raise ValueError(f"{name} evidence and Boolean mask must have shape [T,2]")
        if values.is_floating_point() or values.is_complex():
            raise TypeError(f"{name} evidence must be discrete integer or Boolean data")
        if values.device != states.device or known.device != states.device:
            raise ValueError(f"{name} evidence must share the support device")
        for index, (column, size) in enumerate(zip(columns, domains)):
            selected = values[:, index][known[:, index]]
            if not bool(((selected >= 0) & (selected < size)).all()):
                raise ValueError(f"observed {name} evidence is outside its declared domain")
            compatible = states[:, column][None, :] == values[:, index, None]
            support &= ~known[:, index, None] | compatible
    if not bool(support.any(-1).all()):
        raise GroundingError("communication evidence contradicts fixed threshold support")
    return support


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise GroundingError(f"producer must define exactly one {name}")
    return matches[0]


def _check_assignment(function: ast.FunctionDef, target: str, expression: str) -> None:
    expected = ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False)
    expressions = [node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and any(isinstance(item, ast.Name) and item.id == target for item in node.targets)]
    if len(expressions) != 1 or ast.dump(expressions[0], include_attributes=False) != expected:
        raise GroundingError(f"producer {function.name}.{target} formula differs from declared support")


def load_communication_authority(repo_root: Path | None = None) -> dict:
    """Validate profile, executable strict thresholds and exact ontology roles.

    The L=>P restriction is source-derived for this checked profile only. It is
    never inferred from missing positive samples. Six-decimal source rounding
    leaves a >0.0124 loss margin at the strict latency boundary.
    """
    root = _ROOT if repo_root is None else Path(repo_root)
    profile = json.loads((root / _PROFILE).read_text())
    if profile["profile_id"] != "aeroworld_compute_comm_supplement_v1" or profile["model"]["model_version"] != "1.6.0":
        raise GroundingError("communication profile/model identity differs from declared finite scope")
    channel, flow = profile["communication"]["channel"], profile["communication"]["flow_profiles"]["uav"]
    expected = {"nominal_latency_ms": 20.0, "worst_case_latency_ms": 220.0, "maximum_packet_loss_ratio": 0.25}
    if any(channel[key] != value for key, value in expected.items()) or flow["latency_threshold_ms"] != 120.0 or flow["packet_loss_threshold_ratio"] != 0.05:
        raise GroundingError("communication thresholds/envelope differ from the Q domain derivation")
    contracts = yaml.safe_load((root / _CONTRACT).read_text())
    for predicate in PREDICATE_ORDER:
        if predicate not in contracts["predicate_execution"]["executable_l1_predicate_ids"]:
            raise GroundingError(f"communication predicate is not executable: {predicate}")
        spec = contracts["grounding_specs"][predicate]
        if spec["role_sources"] != {"flow": {"kind": "record_binding", "path": "flow"}}:
            raise GroundingError("communication predicate must retain its official flow-only role")
        if spec["candidate_authority"] != {"kind": "direct_predicate_row", "source": "compute_comm_supplement/predicate_truth.jsonl"}:
            raise GroundingError("communication direct predicate source differs")
        if contracts["predicates"][predicate]["evaluation"] != {"op": "direct_truth", "field": "direct_predicate_value"}:
            raise GroundingError("communication predicate execution contract differs")
    ontology = Graph().parse(root / _ONTOLOGY, format="turtle")
    ontology_roles = {}
    for predicate in PREDICATE_ORDER:
        nodes = list(ontology.subjects(_AW.predicateIdentifier, Literal(predicate)))
        if len(nodes) != 1:
            raise GroundingError(f"ontology must uniquely declare {predicate}")
        roles = list(ontology.objects(nodes[0], _AW.declaresArgumentRole))
        if len(roles) != 1 or ontology.value(roles[0], _AW.argumentRoleKey) != Literal("flow") or ontology.value(roles[0], _WORLD.argumentClass) != _DOM.DataFlow:
            raise GroundingError("communication ontology flow role/class differs")
        ontology_roles[predicate] = {"flow": str(_DOM.DataFlow)}
    tree = ast.parse((root / _PRODUCER).read_text())
    functions = [_function(tree, name) for name in ("_truth_from_latency", "_truth_from_packet_loss", "_link_quality", "_round", "_station_operational_state")]
    for function in functions[:2]:
        comparisons = [node for node in ast.walk(function) if isinstance(node, ast.Compare)
                       and isinstance(node.left, ast.Name) and node.left.id == "value"]
        if len(comparisons) != 1 or len(comparisons[0].ops) != 1 or not isinstance(comparisons[0].ops[0], ast.Gt) or ast.dump(comparisons[0].comparators[0]) != ast.dump(ast.Name(id="threshold", ctx=ast.Load())):
            raise GroundingError("communication predicate thresholds must remain strict >")
    for function, field in zip(functions[:2], ("latency_ms", "packet_loss_ratio")):
        _check_assignment(function, "value", f'row["link_quality"].get("{field}")')
        _check_assignment(function, "threshold", f'row["quality_thresholds"].get("{field}")')
    _check_assignment(functions[2], "latency_ms", "nominal_latency + (1.0 - score) * max(0.0, worst_latency - nominal_latency)")
    _check_assignment(functions[2], "packet_loss_ratio", "max(0.0, min(1.0, (1.0 - score) ** 2 * maximum_packet_loss))")
    round_calls = [node for node in ast.walk(functions[3]) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "round"]
    if len(round_calls) != 1 or len(round_calls[0].args) != 2 or not isinstance(round_calls[0].args[1], ast.Constant) or round_calls[0].args[1].value != 6:
        raise GroundingError("communication source rounding differs from support proof")
    rounding_error = 0.5e-6
    score_upper_when_latency_exceeds = 1 - (
        flow["latency_threshold_ms"] - rounding_error - channel["nominal_latency_ms"]
    ) / (channel["worst_case_latency_ms"] - channel["nominal_latency_ms"])
    loss_lower = channel["maximum_packet_loss_ratio"] * (1 - score_upper_when_latency_exceeds) ** 2 - rounding_error
    implication_margin = loss_lower - flow["packet_loss_threshold_ratio"]
    if implication_margin <= 0:
        raise GroundingError("source rounding does not certify L=>P for this profile")
    # Runtime registry emits world:DataFlow as a producer label. The ontology
    # URI is dom:DataFlow; this explicit alias does not assert URI equality.
    from Dataset.semantic_simulation.compute_comm import ONTOLOGY_PREDICATE_ROLE_CLASSES
    if any(ONTOLOGY_PREDICATE_ROLE_CLASSES[predicate] != {"flow": "world:DataFlow"} for predicate in PREDICATE_ORDER):
        raise GroundingError("runtime flow class label differs from declared ontology alias")
    refs = [f"{_PRODUCER}:{function.lineno}" for function in functions]
    refs += [_PROFILE, _CONTRACT, _ONTOLOGY]
    return {
        "profile_id": profile["profile_id"], "model_version": profile["model"]["model_version"],
        "source_refs": refs, "predicate_order": list(PREDICATE_ORDER),
        "roles": {"actor": "world:UnmannedAircraft", "station": "world:CommunicationStation", "flow": "dom:DataFlow"},
        "runtime_roles": {"actor": "world:UnmannedAircraft", "station": "world:CommunicationStation", "flow": "world:DataFlow"},
        "runtime_class_aliases": {"world:DataFlow": str(_DOM.DataFlow)},
        "predicate_ontology_roles": ontology_roles,
        "state_domains": {"A": [0, 1], "Q": [0, 1, 2]},
        "state_sources": {"A": "station_operational_state.availability", "Q": ["link_quality.latency_ms", "link_quality.packet_loss_ratio", "quality_thresholds"]},
        "A_definition": "1[availability<1]; reduced record context, not station-wide unavailable",
        "Q_definition": ["both pass", "loss only exceeds", "both exceed"],
        "support": "Q->L=1[Q=2], Q->P=1[Q>=1]; all 2x3 A/Q states legal",
        "support_derivation": "latency=20+200*(1-s), loss=.25*(1-s)^2; strict >120 and >.05 imply L=>P",
        "rounded_loss_implication_margin": implication_margin,
        "finite_scope": "exact finite-domain inference, no continuous-dynamics integration claim",
    }


@dataclass(frozen=True)
class MixedFactor:
    factor_id: str
    category: str
    scope: tuple[StateKey, ...]
    source_refs: tuple[str, ...]
    selection_key: str | None = None


@dataclass(frozen=True)
class MixedGraph:
    variables: tuple[Variable, ...]
    factors: tuple[MixedFactor, ...]
    bindings: tuple[RoleBinding, ...]
    ticks: tuple[int, ...]
    authority: dict


def build_mixed_graph(
    *, episode_id: str, branch_id: str, ticks: Sequence[int], bindings: Sequence[RoleBinding],
    source_refs: Sequence[str], initial_joint_state: int,
) -> MixedGraph:
    """Ground fixed and candidate scopes from exact actor/station/flow records.

    A uses the complete producer record owner because operational-state
    production reads actor and station context. Q and official L/P use flow.
    Future entries have no observations; no future source data is accepted.
    """
    authority = load_communication_authority()
    bindings, ticks, source_refs = tuple(bindings), tuple(ticks), tuple(source_refs)
    if tuple((item.role, item.ontology_class) for item in bindings) != tuple(authority["roles"].items()):
        raise GroundingError("mixed graph needs exact ordered actor/station/flow class bindings")
    if any("|" in item.entity_id for item in bindings) or len({item.entity_id for item in bindings}) != 3:
        raise GroundingError("mixed graph role identities must be distinct exact IDs")
    if not 1 <= len(ticks) <= 11 or any(type(tick) is not int or tick < 0 for tick in ticks) or any(b - a != 5 for a, b in zip(ticks, ticks[1:])):
        raise GroundingError("mixed graph must use ordered five-tick adjacent records")
    if not source_refs or any(not isinstance(ref, str) or not ref for ref in source_refs):
        raise GroundingError("mixed graph needs explicit prefix source references")
    compile_mixed_support(initial_joint_state, len(ticks))
    record_owner = "|".join(item.entity_id for item in bindings)
    flow_binding = (bindings[2],)
    fields = STATE_FIELDS + PREDICATE_ORDER
    rows, variables, factors = [], [], []
    authority_refs = tuple(authority["source_refs"])
    for index, tick in enumerate(ticks):
        row = {}
        for column, (name, field) in enumerate(zip(VARIABLE_NAMES, fields)):
            key = StateKey(record_owner if name == "A" else bindings[2].entity_id, field, episode_id, tick, branch_id)
            value = int(DEFAULT_MIXED_ASSIGNMENTS[initial_joint_state, column]) if index == 0 else None
            if value is not None and column >= 2:
                value = bool(value)
            variables.append(Variable(key, VariableKind.STATE if column < 2 else VariableKind.PREDICATE,
                                      value, index == 0, bindings if name == "A" else flow_binding,
                                      source_refs if index == 0 else authority_refs))
            row[name] = key
        rows.append(row)
        for target in ("L", "P"):
            factors.append(MixedFactor(f"threshold_{target}@{tick}", "fixed_rule", (row["Q"], row[target]), authority_refs))
        for name in ("A", "Q"):
            factors.append(MixedFactor(f"unary_{name}@{tick}", "neural_additive", (row[name],), source_refs))
        for target in ("L", "P"):
            name = "A" + target
            factors.append(MixedFactor(f"{name}@{tick}", "neural_candidate", (row["A"], row[target]), source_refs + authority_refs, name))
        if index:
            previous = rows[index - 1]
            for name, left, right in (("AA", "A", "A"), ("QQ", "Q", "Q"),
                                      ("Aprev_Pnext", "A", "P"), ("Lprev_Anext", "L", "A")):
                factors.append(MixedFactor(f"{name}@{ticks[index-1]}:{tick}", "neural_temporal",
                                           (previous[left], row[right]), source_refs + authority_refs, name))
    keys = {variable.key for variable in variables}
    if len(keys) != len(variables) or any(any(key not in keys for key in factor.scope) for factor in factors):
        raise GroundingError("mixed graph factor scopes must use unique shared variable keys")
    return MixedGraph(tuple(variables), tuple(factors), bindings, ticks, authority)

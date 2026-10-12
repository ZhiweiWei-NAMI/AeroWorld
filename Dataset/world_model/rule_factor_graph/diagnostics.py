"""Inspectable association diagnostics for the bounded mixed communication graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch


CROSS_ASSOCIATIONS = ("AL", "AP", "Aprev_Pnext", "Lprev_Anext")


def json_value(value: Any) -> Any:
    """Convert measured tensors and dataclasses without admitting NaN artifacts."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def summarize_coefficients(coefficients: Mapping[str, torch.Tensor]) -> dict:
    result = {}
    for name, tensor in coefficients.items():
        values = tensor.detach().to(dtype=torch.float64)
        if values.numel() == 0 or not bool(torch.isfinite(values).all()):
            raise ValueError(f"Coefficient {name} is empty or non-finite")
        result[name] = {
            "shape": list(values.shape),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "rms": float(values.square().mean().sqrt()),
            "std_population": float(values.std(unbiased=False)),
            "nonzero_count": int(torch.count_nonzero(values)),
        }
    return result


def association_knockouts(
    model: torch.nn.Module,
    history: torch.Tensor,
    branch_features: torch.Tensor,
    posterior: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    legal_support: torch.Tensor,
) -> dict:
    """Measure one-window local scores and joint posterior after each removal.

    ``posterior`` uses the same fixed evidence support for every call. A local
    removal may leave correlations through the other factors in the graph.
    """
    if history.shape[0] != 1 or branch_features.shape[0] != 1:
        raise ValueError("Knockout diagnostics require exactly one batched window")
    model.eval()
    with torch.no_grad():
        baseline = model(history, branch_features)
        base_posterior = posterior(baseline.unary[0], baseline.transitions[0])
        if legal_support.shape != baseline.unary.shape[1:]:
            raise ValueError("Legal support differs from single-window score shape")
        outputs = {}
        for removed in [(name,) for name in CROSS_ASSOCIATIONS] + [CROSS_ASSOCIATIONS]:
            score = model(history, branch_features, knockout=removed)
            changed_posterior = posterior(score.unary[0], score.transitions[0])
            unary_difference = (baseline.unary[0] - score.unary[0])[legal_support]
            transition_difference = baseline.transitions[0] - score.transitions[0]
            posterior_difference = base_posterior - changed_posterior
            outputs["+".join(removed)] = {
                "removed": list(removed),
                "legal_unary_max_abs_change": float(unary_difference.abs().max()),
                "transition_max_abs_change": float(transition_difference.abs().max()),
                "joint_posterior_max_abs_change": float(posterior_difference.abs().max()),
                "joint_posterior_l1_change": float(posterior_difference.abs().sum()),
                "posterior": changed_posterior,
                "coefficient_summary": summarize_coefficients(score.coefficients),
            }
        return {
            "baseline_posterior": base_posterior,
            "baseline_coefficient_summary": summarize_coefficients(baseline.coefficients),
            "effective_coefficients": baseline.coefficients,
            "raw_coefficients_before_fixed_structure_mask": baseline.raw_coefficients,
            "knockouts": outputs,
            "interpretation": "Association removal is a model counterfactual, with no causal claim.",
        }


def export_mixed_graph(
    *,
    episode_id: str,
    owner_id: str,
    roles: Mapping[str, str],
    ticks: Sequence[int],
    cutoff_tick: int,
    branch_count: int,
    source_refs: Sequence[str],
    structure: str,
    coefficient_summary: Mapping[str, Any],
    basis_diagnostics: Mapping[str, Any],
    canonical_graph: Any,
    conditioned_inputs: Mapping[str, Any],
) -> dict:
    """Export actual factor scopes; neural conditioner tensors are not entities."""
    if set(roles) != {"actor", "station", "flow"}:
        raise ValueError("Communication graph requires exact actor/station/flow roles")
    if owner_id != "|".join(roles[role] for role in ("actor", "station", "flow")):
        raise ValueError("Graph owner differs from exact typed role bindings")
    if structure not in ("none", "AL", "AP", "AL_AP"):
        raise ValueError("Unknown fixed candidate structure")
    if not ticks or ticks[0] != cutoff_tick or branch_count < 1:
        raise ValueError("Graph must start at the cutoff and have shared motion branches")
    selected = {"AL": structure in ("AL", "AL_AP"), "AP": structure in ("AP", "AL_AP")}
    nodes, edges = [], []
    fixed_factors, candidate_factors, learned_factors = [], [], []

    def variable_id(name: str, time: int) -> str:
        return f"{name}:{time}"

    def factor(name: str, scope: list[tuple[str, int]], kind: str, **extra: Any) -> None:
        identity = f"factor:{name}:{scope[0][1]}"
        entry = {"id": identity, "node_type": "factor", "factor_kind": kind,
                 "name": name, "owner_id": owner_id, "scope": [variable_id(*item) for item in scope],
                 "source_refs": list(source_refs), **extra}
        nodes.append(entry)
        for name_in_scope, time in scope:
            edges.append({"source": variable_id(name_in_scope, time), "target": identity,
                          "edge_class": kind + "_incidence"})
        if kind == "hard_rule":
            fixed_factors.append(entry)
        elif kind == "candidate_association":
            candidate_factors.append(entry)
            if extra["selected"]:
                learned_factors.append(entry)
        else:
            learned_factors.append(entry)

    domains = {"A": [0, 1], "Q": ["both_pass", "loss_only", "both_fail"],
               "L": [False, True], "P": [False, True]}
    meanings = {"A": "availability < 1; reduced availability, not station down",
                "Q": "finite threshold class on the verified common-score profile",
                "L": "latency strictly exceeds declared limit",
                "P": "packet loss strictly exceeds declared limit"}
    for time, tick in enumerate(ticks):
        for name in ("A", "Q", "L", "P"):
            nodes.append({"id": variable_id(name, time), "node_type": "variable",
                          "variable_kind": "state" if name in ("A", "Q") else "predicate",
                          "name": name, "meaning": meanings[name], "domain": domains[name],
                          "episode_id": episode_id,
                          "owner_id": owner_id if name == "A" else roles["flow"],
                          "roles": dict(roles) if name == "A" else {"flow": roles["flow"]},
                          "tick": tick, "future_latent": tick > cutoff_tick,
                          "observations": "cutoff prefix only; future labels are separate supervision"})
        factor("Q_to_L", [("Q", time), ("L", time)], "hard_rule",
               fixed_mapping=[False, False, True], gateable=False)
        factor("Q_to_P", [("Q", time), ("P", time)], "hard_rule",
               fixed_mapping=[False, True, True], gateable=False)
        factor("A_unary", [("A", time)], "neural_main_effect", coefficient_keys=["A"])
        factor("Q_unary", [("Q", time)], "neural_main_effect", coefficient_keys=["L", "P"])
        for name, predicate in (("AL", "L"), ("AP", "P")):
            factor(name, [("A", time), (predicate, time)], "candidate_association",
                   selected=selected[name], selection="fixed structure enumeration",
                   coefficient_key=name, candidate_reason="exact typed roles and shared state owner")
        if time + 1 < len(ticks):
            for name, left, right in (("AA", "A", "A"), ("QQ", "Q", "Q"),
                                      ("Aprev_Pnext", "A", "P"), ("Lprev_Anext", "L", "A")):
                factor(name, [(left, time), (right, time + 1)], "neural_temporal_association",
                       coefficient_key=name, candidate_reason="same owner and adjacent producer records")
    from .mixed_graph import STATE_FIELDS, PREDICATE_ORDER
    field_by_name = dict(zip(("A", "Q", "L", "P"), (*STATE_FIELDS, *PREDICATE_ORDER)))
    canonical_variables = {(variable.key.owner_id, variable.key.field, variable.key.tick): variable
                           for variable in canonical_graph.variables}
    by_id = {node["id"]: node for node in nodes if node["node_type"] == "variable"}
    actual_variable_signatures = set()
    for node in by_id.values():
        signature = (node["owner_id"], field_by_name[node["name"]], node["tick"])
        actual_variable_signatures.add(signature)
        variable = canonical_variables[signature]
        node.update({"typed_key": asdict(variable.key), "observed": variable.observed,
                     "value": variable.value, "role_bindings": [asdict(binding) for binding in variable.role_bindings],
                     "source_refs": list(variable.source_refs)})
    if actual_variable_signatures != set(canonical_variables):
        raise ValueError("Diagnostic variables differ from the canonical typed graph")
    category = {"hard_rule": "fixed_rule", "neural_main_effect": "neural_additive",
                "candidate_association": "neural_candidate", "neural_temporal_association": "neural_temporal"}
    canonical_factors = {(factor.category, tuple((key.owner_id, key.field, key.tick) for key in factor.scope)): factor
                         for factor in canonical_graph.factors}
    actual_factor_signatures = set()
    for node in nodes:
        if node["node_type"] != "factor":
            continue
        scope = tuple((by_id[identity]["owner_id"], field_by_name[by_id[identity]["name"]], by_id[identity]["tick"])
                      for identity in node["scope"])
        signature = (category[node["factor_kind"]], scope)
        actual_factor_signatures.add(signature)
        canonical = canonical_factors[signature]
        node.update({"canonical_factor_id": canonical.factor_id,
                     "typed_scope": [asdict(key) for key in canonical.scope],
                     "source_refs": list(canonical.source_refs)})
    if actual_factor_signatures != set(canonical_factors):
        raise ValueError("Diagnostic factor scopes differ from the canonical typed graph")
    node_ids = {node["id"] for node in nodes}
    if len(node_ids) != len(nodes) or any(edge["source"] not in node_ids or edge["target"] not in node_ids for edge in edges):
        raise ValueError("Export contains duplicate nodes or dangling factor incidence")
    return {
        "purpose": "bounded mixed joint prediction; associations are not causal edges",
        "episode_id": episode_id, "owner_id": owner_id, "roles": dict(roles),
        "cutoff_tick": cutoff_tick, "ticks": list(ticks), "branch_count": branch_count,
        "motion_contract": "one frozen shared sample per entity/field/time in each branch; equal probability mixture",
        "nodes": nodes, "edges": edges, "fixed_factors": fixed_factors,
        "candidate_factors": candidate_factors, "learned_factors": learned_factors,
        "selected_structure": structure, "coefficient_summary": coefficient_summary,
        "feature_design": basis_diagnostics, "source_refs": list(source_refs),
        "canonical_scope_and_source_crosscheck": True,
        "conditioned_inputs": conditioned_inputs,
        "conditioning_interpretation": "Available history and sampled X condition coefficient functions; factor interaction scopes are the A/Q/L/P variables listed in each potential.",
        "finite_model_exactness": "Exact partition and latent marginalization apply to six discrete states, not continuous dynamics.",
    }

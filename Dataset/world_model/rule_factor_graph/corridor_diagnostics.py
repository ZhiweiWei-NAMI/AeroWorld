"""Measured learning and scope evidence for the bounded corridor experiment."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return "infinity" if value > 0 else "negative_infinity" if value < 0 else "nan"
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def coefficient_summary(coefficients: dict[str, torch.Tensor]) -> dict:
    return {name: {"shape": list(tensor.shape), "mean": float(tensor.mean()),
                   "min": float(tensor.min()), "max": float(tensor.max()),
                   "std": float(tensor.std(unbiased=False))}
            for name, tensor in coefficients.items()}


def event_gradient_evidence(model: torch.nn.Module, loss_function: Callable,
                            score_function: Callable, *, window_ids: list[str],
                            epsilon: float = 1e-5) -> dict:
    """Differentiate the same mixture event NLL used by the optimizer."""
    model.eval()
    model.zero_grad(set_to_none=True)
    scores = score_function(model)
    loss = loss_function(scores)
    objective = loss.event_nll.mean()
    objective.backward()
    gradients = {name: float(parameter.grad.norm()) if parameter.grad is not None else None
                 for name, parameter in model.named_parameters()}
    candidates = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())]
    selected = []
    for prefix in ("history_encoder", "b_context", "i_context", "b_emission_head",
                   "i_emission_head", "b_persistence_head", "i_persistence_head", "gamma_head"):
        module = [(name, parameter) for name, parameter in candidates if name.startswith(prefix + ".")]
        if module:
            selected.append(max(module, key=lambda item: float(item[1].grad.abs().max())))
    checked = []
    for name, parameter in selected:
        index = int(parameter.grad.abs().reshape(-1).argmax())
        analytic = float(parameter.grad.reshape(-1)[index])
        with torch.no_grad():
            original = parameter.reshape(-1)[index].clone()
            parameter.reshape(-1)[index] = original + epsilon
            upper = loss_function(score_function(model)).event_nll.mean()
            parameter.reshape(-1)[index] = original - epsilon
            lower = loss_function(score_function(model)).event_nll.mean()
            parameter.reshape(-1)[index] = original
        finite = float((upper - lower) / (2 * epsilon))
        checked.append({"parameter": name, "flat_index": index, "value": float(original),
                        "analytic_gradient": analytic, "finite_difference": finite,
                        "absolute_error": abs(analytic - finite)})
    if not checked or max(item["absolute_error"] for item in checked) > 1e-6:
        raise AssertionError("The actual mixture event gradient does not match finite differences")
    if not any(value is not None and value > 1e-10 for name, value in gradients.items() if "encoder" in name):
        raise AssertionError("Actual event NLL has no measured history-encoder gradient")
    if not any(value is not None and value > 1e-10 for name, value in gradients.items() if "gamma_head" in name):
        raise AssertionError("Actual event NLL has no measured association gradient")
    branch = loss.branch_event_log_probability.detach()
    finite_branch = torch.isfinite(branch).all(dim=1)
    mean_branch_loss = (-branch[finite_branch]).mean() if bool(finite_branch.any()) else None
    return {"window_ids": window_ids, "objective": "mean of negative log mean_k normalized joint event probability",
            "event_nll": float(objective.detach()), "epsilon": epsilon,
            "finite_differences": checked, "parameter_gradient_norms": gradients,
            "branch_event_log_probability": branch,
            "branch_responsibility": loss.branch_responsibility,
            "responsibility_row_sum_error": float((loss.branch_responsibility.sum(-1) - 1).abs().max()),
            "mean_branch_nll_if_all_finite": None if mean_branch_loss is None else float(mean_branch_loss),
            "gradient_sign": "NLL: responsibility-weighted E_full[basis] - E_event[basis]"}


def parameter_changes(model: torch.nn.Module, initial: dict[str, torch.Tensor]) -> list[dict]:
    records = []
    for name, parameter in model.named_parameters():
        before = initial[name]
        delta = parameter.detach().cpu() - before
        index = int(delta.abs().reshape(-1).argmax())
        if any(part in name for part in ("encoder", "gamma", "persistence", "emission")):
            records.append({"parameter": name, "flat_index": index,
                            "before": float(before.reshape(-1)[index]),
                            "after": float(parameter.detach().reshape(-1)[index]),
                            "change": float(delta.reshape(-1)[index]),
                            "tensor_change_l2": float(delta.norm())})
    return records


def export_corridor_graph(query: Any, *, parameter_manifest: dict,
                          coefficients: dict[str, torch.Tensor], branches: int) -> dict:
    """Typed owner scopes distinguish entities, latent state, rules and parameters."""
    nodes, edges = [], []
    entities = [{"id": query.actor_id, "node_type": "entity", "type": "world:UnmannedAircraft"}]
    entities += [{"id": corridor, "node_type": "entity", "type": "world:AirspaceCorridor"}
                 for corridor in query.corridor_ids]
    for entity in entities:
        nodes.append(entity)
    sources = list(query.prefix_source_refs) + list(query.geometry_source_refs)
    authority_sources = ["Dataset/world_model/rule_factor_graph/manifest.json",
                         "Dataset/semantic_rules/predicates/world_truth_predicate_contracts.yaml",
                         "Dataset/knowledge_graph/domain/airspace_operations.ttl",
                         "Dataset/semantic_simulation/predicate_state_computers.py#_point_in_oriented_corridor_box"]
    from .corridor_state_space import geometry_membership_support
    geometric_support = geometry_membership_support(*query.boxes).tolist()
    for index, tick in enumerate(query.ticks):
        variables = [("B", query.actor_id, "B")]
        for corridor_index, corridor in enumerate(query.corridor_ids, start=1):
            variables += [(f"I{corridor_index}", f"{query.actor_id}|{corridor}", "I")]
            variables += [(f"{name}{corridor_index}", f"{query.actor_id}|{corridor}", name)
                          for name in ("O", "M", "S")]
        for name, owner, field in variables:
            identity = f"{name}:{tick}"
            nodes.append({"id": identity, "node_type": "variable",
                          "variable_kind": "latent_state" if name[0] in "BI" else "predicate",
                          "name": name, "typed_key": {"owner_id": owner, "episode_id": query.episode_id,
                                                       "tick": tick, "field": field},
                          "domain": [0, 1], "observed": index == 0,
                          "observed_prefix_refs": list(query.prefix_source_refs) if index == 0 else [],
                          "conditioning_source_refs": sources,
                          "regime_definition": "true speed_mps > fixed episode tau_mps" if name == "B" else None})
            owners = [query.actor_id] if name == "B" else [query.actor_id, query.corridor_ids[int(name[-1])-1]]
            edges += [{"source": owner_id, "target": identity, "edge_class": "entity_ownership"} for owner_id in owners]
            if name == "B":
                edges.append({"source": "parameter:tau", "target": identity, "edge_class": "fixed_regime_definition"})
        factor_id = f"rule:geometry:{tick}"
        scope_ids = [f"I1:{tick}", f"I2:{tick}"]
        nodes.append({"id": factor_id, "node_type": "factor", "factor_kind": "hard_geometry",
                      "scope": scope_ids, "feasible_I1_I2": geometric_support,
                      "source_refs": list(query.geometry_source_refs)+authority_sources[-1:], "gateable": False})
        edges += [{"source": item, "target": factor_id, "edge_class": "factor_incidence"} for item in scope_ids]
        for corridor_index in (1, 2):
            for name, scope, formula in (("O", [f"I{corridor_index}"], "O=I"),
                                        ("M", [f"I{corridor_index}", "B"], "M=I&B"),
                                        ("S", [f"I{corridor_index}", "B"], "S=I&notB")):
                factor_id = f"rule:{name}{corridor_index}:{tick}"
                scope_ids = [f"{item}:{tick}" for item in (*scope, f"{name}{corridor_index}")]
                nodes.append({"id": factor_id, "node_type": "factor", "factor_kind": "hard_rule",
                              "scope": scope_ids, "formula": formula, "log_values": [0, "negative_infinity"],
                              "gateable": False, "source_refs": authority_sources})
                edges += [{"source": item, "target": factor_id, "edge_class": "factor_incidence"} for item in scope_ids]
            for name, scope, formula in ((f"I{corridor_index}_emission", [f"I{corridor_index}"], "score(I) from signed-distance/history MLP"),
                                        (f"gamma{corridor_index}", ["B", f"I{corridor_index}"], "gamma*(1-B)*I")):
                factor_id = f"neural:{name}:{tick}"
                scope_ids = [f"{item}:{tick}" for item in scope]
                nodes.append({"id": factor_id, "node_type": "factor", "factor_kind": "neural_predictive",
                              "scope": scope_ids, "formula": formula, "coefficient": name,
                              "source_refs": sources, "causal_claim": False})
                edges += [{"source": item, "target": factor_id, "edge_class": "factor_incidence"} for item in scope_ids]
        factor_id = f"neural:B_emission:{tick}"
        nodes.append({"id": factor_id, "node_type": "factor", "factor_kind": "neural_predictive",
                      "scope": [f"B:{tick}"], "formula": "score(B) from shared sampled speed/history MLP",
                      "occurrences_per_actor_time": 1, "source_refs": sources})
        edges.append({"source": f"B:{tick}", "target": factor_id, "edge_class": "factor_incidence"})
        if index:
            for name in ("B", "I1", "I2"):
                scope = [f"{name}:{query.ticks[index - 1]}", f"{name}:{tick}"]
                factor_id = f"neural:persistence:{name}:{tick}"
                nodes.append({"id": factor_id, "node_type": "factor", "factor_kind": "neural_temporal",
                              "scope": scope, "formula": "softplus(lambda_context)*1[z_previous=z_current]",
                              "source_refs": sources})
                edges += [{"source": item, "target": factor_id, "edge_class": "factor_incidence"} for item in scope]
    nodes.append({"id": "parameter:tau", "node_type": "parameter", "name": "episode tau_mps",
                  "value": query.tau_mps, "trainable": False, "source_refs": sources})
    for name, tensor in parameter_manifest["parameter_tensors"].items():
        nodes.append({"id": "parameter:"+name, "node_type": "parameter", "name": name,
                      "shape": tensor["shape"], "trainable": tensor["trainable"]})
    for index, source in enumerate(dict.fromkeys(sources+authority_sources)):
        provenance_id = f"provenance:{index}"
        nodes.append({"id": provenance_id, "node_type": "provenance", "source_ref": source})
        edges += [{"source": provenance_id, "target": node["id"], "edge_class": "provenance"}
                  for node in nodes if source in node.get("source_refs", [])]
    for node in nodes:
        if node["node_type"] == "factor" and node["factor_kind"].startswith("neural"):
            for name in parameter_manifest["parameter_tensors"]:
                edges.append({"source": "parameter:"+name, "target": node["id"], "edge_class": "neural_parameter_conditioning"})
    return {"episode_id": query.episode_id, "actor_id": query.actor_id,
            "corridor_ids": query.corridor_ids, "cutoff_tick": query.cutoff_tick,
            "branches": branches, "entities": entities, "nodes": nodes, "edges": edges,
            "boxes": query.boxes, "parameter_manifest": parameter_manifest,
            "coefficient_summary": coefficient_summary(coefficients),
            "sample_contract": "One actor trajectory per branch is shared by both corridors; sampled forecast is evidence",
            "latent_contract": "B is one actor/time speed regime; I1/I2 are unknown true future occupancy",
            "partition_contract": "Eight latent states per slice; many-to-one predicate mapping retains both outside B states",
            "graph_representation": "One branch latent graph template; actual cached shared forecast evidence is listed in q_manifest.json",
            "architecture_scope": "Domain-specific B/I feature encoder and separate B/I scoring modules; unified typed-variable encoder and shared scope scorer are not implemented",
            "associations": "Predictive hypotheses supported by exact typed actor-state owner; no causal claim"}

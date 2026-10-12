"""Hard AST support, preserving the authoritative four-valued evaluator."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from Dataset.semantic_truth.world_truth import (
    MISSING, _evaluate_expression, _expression_truth_value, _missing_value,
    _value_at_path,
)
from .unified_contracts import (
    ContractError, PredicateContract, SUPPORTED_OPERATORS,
    UnsupportedOperatorError, expression_fields, walk_expression,
)


def set_field(context: dict[str, Any], field: str, value: Any) -> None:
    parts = field.split(".")
    current = context
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        if not isinstance(current[part], dict):
            raise ContractError(f"field path collision: {field}")
        current = current[part]
    current[parts[-1]] = value


def _validate_ast(spec: Mapping[str, Any]) -> None:
    for node in walk_expression(spec):
        op = node["op"]
        if op not in SUPPORTED_OPERATORS:
            raise UnsupportedOperatorError(f"unsupported evaluation operator: {op!r}")
        if op == "all":
            if not isinstance(node.get("args"), (list, tuple)) or not node["args"]:
                raise ContractError("all requires nonempty explicit args")
        elif op in {"gt_fields", "absolute_difference_gt"}:
            if not all(isinstance(node.get(name), str) and node[name] for name in ("left", "right")):
                raise ContractError(f"{op} requires explicit left/right field paths")
        elif not isinstance(node.get("field"), str) or not node["field"]:
            raise ContractError(f"{op} requires an explicit field path")
        for name in (("threshold",) if op in {"gt", "gte", "lt", "lte", "absolute_difference_gt"}
                     else ("minimum", "maximum") if op == "outside_range" else ()):
            if not isinstance(node.get(name), str) or not node[name]:
                raise ContractError(f"{op} requires an explicit parameter {name}")
        if op in {"equals", "gt_constant"} and "value" not in node:
            raise ContractError(f"{op} requires an explicit constant")
        if op == "in" and not isinstance(node.get("values"), list):
            raise ContractError("in requires an explicit values list")


@dataclass(frozen=True)
class CompiledRule:
    contract: PredicateContract
    previous_fields: tuple[str, ...]

    def evaluate(self, context: Mapping[str, Any], previous_context: Mapping[str, Any] | None = None,
                 parameters: Mapping[str, Any] | None = None, *, applicable: bool = True,
                 source_out_of_scope: bool = False) -> str:
        if not self.contract.executable:
            raise ContractError(f"non-executable predicate: {self.contract.predicate_id}")
        if type(applicable) is not bool or type(source_out_of_scope) is not bool:
            raise ContractError("applicability and declared source out-of-scope status must be booleans")
        if not applicable or source_out_of_scope:
            return "out_of_scope"
        if any(_missing_value(_value_at_path(context, field.field))
               for field in self.contract.required_fields):
            return "unknown"
        parameters = {} if parameters is None else parameters
        absent = set(self.contract.threshold_parameters) - set(parameters)
        if absent:
            raise ContractError(f"missing governed threshold parameters: {sorted(absent)}")
        previous_context = {} if previous_context is None else previous_context
        return _expression_truth_value(_evaluate_expression(
            self.contract.evaluation_spec, context, previous_context, parameters))

    def log_support(self, output_value: str, context: Mapping[str, Any],
                    previous_context: Mapping[str, Any] | None = None,
                    parameters: Mapping[str, Any] | None = None, **kwargs) -> float:
        if output_value not in {"false", "true", "unknown", "out_of_scope"}:
            raise ContractError(f"invalid truth assignment: {output_value!r}")
        return 0.0 if output_value == self.evaluate(
            context, previous_context, parameters, **kwargs) else -math.inf


def compile_rule(contract: PredicateContract) -> CompiledRule:
    if not contract.executable:
        raise ContractError(f"cannot compile non-executable predicate: {contract.predicate_id}")
    _validate_ast(contract.evaluation_spec)
    referenced = set(expression_fields(contract.evaluation_spec))
    declared = {field.field for field in contract.required_fields}
    if not referenced <= declared:
        raise ContractError(f"AST references undeclared fields: {sorted(referenced - declared)}")
    previous = tuple(dict.fromkeys(node["field"] for node in walk_expression(contract.evaluation_spec)
                                   if node["op"] == "changed"))
    return CompiledRule(contract, previous)

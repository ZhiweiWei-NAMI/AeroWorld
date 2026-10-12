"""Phase-1 typed corridor rules with exact shared-state grounding."""
from .contracts import (
    BranchState, EdgeProvenance, FactorTemplate, GroundFactor, GroundedSlice,
    GroundingError, PredicateEvidence, RoleBinding, StateKey,
    UnsupportedGroundingError, Variable, VariableKind,
)
from .grounding import PREDICATE_ORDER, ground_corridor_stop_slice, load_manifest, validate_grounded_slice

__all__ = [
    "BranchState", "EdgeProvenance", "FactorTemplate", "GroundFactor", "GroundedSlice",
    "GroundingError", "PredicateEvidence", "RoleBinding", "StateKey",
    "UnsupportedGroundingError", "Variable", "VariableKind",
    "PREDICATE_ORDER", "ground_corridor_stop_slice", "load_manifest", "validate_grounded_slice",
]

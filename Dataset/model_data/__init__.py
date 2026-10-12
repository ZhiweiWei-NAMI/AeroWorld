"""Split-neutral AERO model-data projections and graph interfaces."""

from .action_features import ACTION_FEATURE_DIM, ActionFeatureStore
from .branch_dataset import (
    BranchGraphWindowDataset,
    BranchGraphWindowSample,
    BranchWindowRecord,
)
from .branch_graph import (
    BRANCH_ACTION_FEATURE_CONTRACT,
    BranchActionFeatureProvider,
    BranchGraphError,
    BranchGraphEvidenceStore,
    BranchGraphSequence,
    BranchGraphSequenceStore,
    compose_branch_graph_sequence,
)

from .action_contract import (
    ACTION_SCHEMA_NAME,
    ACTION_SCHEMA_VERSION,
    ActionCommandRecord,
    ActionProjectionDocument,
    OutcomeRecord,
    ProcessRecord,
    TargetBindingRecord,
    serialize_origin_model_input,
)
from .contract import (
    DATASET_SCHEMA_NAME,
    DATASET_SCHEMA_VERSION,
    FORMAL_TICKS,
    INACTIVE_PREDICATE_CODE,
    PREDICATE_TRUTH_CODES,
)
from .dataset import GraphWindow, ObserverBundle, UnifiedGraphFrame

__all__ = [
    "ACTION_FEATURE_DIM",
    "ActionFeatureStore",
    "BRANCH_ACTION_FEATURE_CONTRACT",
    "BranchActionFeatureProvider",
    "BranchGraphError",
    "BranchGraphEvidenceStore",
    "BranchGraphSequence",
    "BranchGraphSequenceStore",
    "BranchGraphWindowDataset",
    "BranchGraphWindowSample",
    "BranchWindowRecord",
    "compose_branch_graph_sequence",
    "ACTION_SCHEMA_NAME",
    "ACTION_SCHEMA_VERSION",
    "ActionCommandRecord",
    "ActionProjectionDocument",
    "DATASET_SCHEMA_NAME",
    "DATASET_SCHEMA_VERSION",
    "FORMAL_TICKS",
    "GraphWindow",
    "INACTIVE_PREDICATE_CODE",
    "ObserverBundle",
    "OutcomeRecord",
    "PREDICATE_TRUTH_CODES",
    "ProcessRecord",
    "TargetBindingRecord",
    "UnifiedGraphFrame",
    "serialize_origin_model_input",
]

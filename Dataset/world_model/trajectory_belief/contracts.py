"""Array and persistence contract shared by the data, sampler and model layers."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Dataset.world_model.transition_prediction.data import LABEL_VIEW, validate_targets

HISTORY_STEPS = 21
FUTURE_STEPS = 10
STEP_TICKS = 5
TICK_SECONDS = 0.1
STEP_SECONDS = STEP_TICKS * TICK_SECONDS
RULE_PARAMETER_NAMES = ("metric_code", "horizontal_limit_m", "vertical_limit_m", "margin_m")


@dataclass
class WindowDataset:
    """Physical arrays are xyz/vxyz; IDs, labels and provenance stay in rows.

    Every history is complete. Missing future states are NaN and are accompanied
    by a false future_present value. Future arrays and masks are supervision,
    never query features. Euclidean H/V equal the governed margin; anisotropic
    H/V are the observed horizontal and vertical limits.
    """

    history: np.ndarray
    future: np.ndarray
    future_present: np.ndarray
    rows: pd.DataFrame
    rule_parameters: np.ndarray
    report: dict = field(default_factory=dict)

    def __post_init__(self):
        self.history = np.asarray(self.history, dtype=np.float64)
        self.future = np.asarray(self.future, dtype=np.float64)
        self.future_present = np.asarray(self.future_present, dtype=bool)
        self.rule_parameters = np.asarray(self.rule_parameters, dtype=np.float64)
        self.rows = self.rows.reset_index(drop=True).copy()
        n = len(self.rows)
        for name, shape in (("history", (n, HISTORY_STEPS, 2, 6)),
                            ("future", (n, FUTURE_STEPS, 2, 6)),
                            ("future_present", (n, FUTURE_STEPS, 2)),
                            ("rule_parameters", (n, 4))):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(self.history).all() or not np.isfinite(self.rule_parameters).all():
            raise ValueError("history and static rule parameters must be finite")
        if not np.isfinite(self.future[self.future_present]).all():
            raise ValueError("present future state must be finite")
        if not np.isnan(self.future[~self.future_present]).all():
            raise ValueError("absent future states must be explicitly NaN")
        if not np.isin(self.rule_parameters[:, 0], [0, 1]).all() or (self.rule_parameters[:, 1:] <= 0).any():
            raise ValueError("invalid metric or nonpositive separation limits")
        validate_targets(self.rows)
        if "row_id" in self.rows and self.rows.row_id.duplicated().any():
            raise ValueError("row IDs must be unique")
        if "current" in self.rows and not self.rows.current.isin([0, 1]).all():
            raise ValueError("initial current truth must be binary 0/1")
        if "state_complete" in self.rows and not np.array_equal(
                self.rows.state_complete.to_numpy(dtype=bool), self.future_present.all(axis=(1, 2))):
            raise ValueError("state_complete disagrees with future_present")

    def __len__(self):
        return len(self.rows)

    def subset(self, indices):
        index = np.arange(len(self))[indices]
        index = np.atleast_1d(index)
        return WindowDataset(self.history[index], self.future[index], self.future_present[index],
                             self.rows.iloc[index], self.rule_parameters[index], self.report.copy())

    def save(self, directory):
        """Write a new isolated artifact directory; existing artifacts are errors."""
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(f"refusing to replace dataset directory: {directory}")
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "windows.npz", history=self.history, future=self.future,
                            future_present=self.future_present, rule_parameters=self.rule_parameters)
        self.rows.to_parquet(directory / "rows.parquet", index=False)
        manifest = dict(array_contract="trajectory-window-v1", label_view=LABEL_VIEW,
                        history_steps=HISTORY_STEPS, future_steps=FUTURE_STEPS,
                        step_ticks=STEP_TICKS, tick_seconds=TICK_SECONDS,
                        state_fields=["x", "y", "z", "vx", "vy", "vz"],
                        rule_parameter_names=list(RULE_PARAMETER_NAMES), report=self.report)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        expected = dict(array_contract="trajectory-window-v1", label_view=LABEL_VIEW,
                        history_steps=HISTORY_STEPS, future_steps=FUTURE_STEPS,
                        step_ticks=STEP_TICKS, tick_seconds=TICK_SECONDS,
                        state_fields=["x", "y", "z", "vx", "vy", "vz"],
                        rule_parameter_names=list(RULE_PARAMETER_NAMES))
        for name, value in expected.items():
            if manifest[name] != value:
                raise ValueError(f"saved dataset contract mismatch: {name}")
        with np.load(directory / "windows.npz", allow_pickle=False) as arrays:
            return cls(history=arrays["history"], future=arrays["future"],
                       future_present=arrays["future_present"], rule_parameters=arrays["rule_parameters"],
                       rows=pd.read_parquet(directory / "rows.parquet"), report=manifest["report"])


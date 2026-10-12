"""Graph-only inference; labels and source trace are not accepted by this interface."""
from __future__ import annotations

import json
import joblib
import numpy as np
import torch

from .data import OUT, GraphDataset, write_json, LEGACY_LABEL_VIEW
from .experiment import engineered, tree_predict
from .model import build_model


class GraphSwitchPredictor:
    """Load a named frozen forecaster, or the candidate selected before test evaluation."""

    def __init__(self, name=None, device="cpu"):
        self.config = json.loads((OUT / "config.json").read_text())
        self.vocab = json.loads((OUT / "vocabulary.json").read_text())
        self.name = json.loads((OUT / "evaluation_protocol.json").read_text())["candidate"] if name is None else name
        self.device = device
        self.models = []
        if self.name in ("GBT", "GBT_primary"):
            self.tree = joblib.load(OUT / "models" / (self.name+".joblib"))
        else:
            name = self.name.removesuffix("_ensemble")
            mode = self.config["run_modes"][name]
            for seed in self.config["optimization_seeds"]:
                model = build_model(self.vocab, mode, self.config["width"]).to(device)
                saved = torch.load(OUT / "models" / f"{name}_seed{seed}.pt", map_location=device, weights_only=False)
                model.load_state_dict(saved["state"])
                model.eval()
                self.models.append(model)

    def __call__(self, graph):
        if self.models:
            x = {k: v.to(self.device) for k, v in graph.items()}
            with torch.no_grad():
                repeats = np.stack([m(x).double().exp().cpu().numpy() for m in self.models])
            probability = repeats.mean(0)
            variability = repeats[:, :, :10].sum(-1).std(0, ddof=1)
        else:
            features = engineered({**graph, "predicate_vocab_size": torch.tensor(len(self.vocab["predicates"]))})
            probability = tree_predict(self.tree, features)
            variability = None
        probability = probability.astype(np.float64)
        probability /= probability.sum(-1, keepdims=True)
        switch = probability[:, :10].sum(-1)
        expected_tick = np.divide(probability[:, :10] @ (np.arange(1, 11)*5), switch,
            out=np.full(len(switch), np.nan), where=switch > 0)
        query = graph["query"]
        current = graph["truth"][-1, query].cpu().numpy()
        assert np.isin(current, [1, 2]).all()
        types = graph["predicate_type"][query].cpu().numpy()
        primary = np.array([self.vocab["predicates"][i] in self.config["primary_predicates"] for i in types])
        return dict(probabilities=probability, switch_probability=switch, no_switch_probability=probability[:, 10],
            cumulative_switch_probability=probability[:, :10].cumsum(-1),
            first_direction=np.where(current == 1, "false_to_true", "true_to_false"),
            expected_switch_tick_given_switch=expected_tick, predicted_bin=probability.argmax(-1),
            predicted_bin_probability=probability.max(-1), optimization_seed_sd=variability,
            supported_predicate=(types > 0) & (primary if "primary" in self.name else True),
            primary_evaluation_task=primary)


def example():
    torch.set_num_threads(4)
    dataset = GraphDataset(label_view=LEGACY_LABEL_VIEW)
    graph, _, trace = dataset.get(("L1-4_v1", 100))
    predictor = GraphSwitchPredictor()
    result = predictor(graph)
    i = int(np.flatnonzero(result["primary_evaluation_task"])[0])
    row = trace.iloc[i]
    item = dict(model=predictor.name, input_contract=predictor.config["input_contract"],
        source_profile="L1-4_v1", source_seed="seed00", cutoff_tick=100,
        input_shapes={key: list(value.shape) for key, value in graph.items()},
        query=dict(predicate=row.predicate, bindings=json.loads(row.bindings_json), current_truth=bool(row.current)),
        first_direction=str(result["first_direction"][i]), bin_width_ticks=5, horizon_ticks=50,
        first_switch_bin_probabilities=result["probabilities"][i, :10].tolist(),
        no_switch_probability=float(result["no_switch_probability"][i]),
        switch_probability=float(result["switch_probability"][i]),
        expected_switch_offset_ticks_given_switch=float(result["expected_switch_tick_given_switch"][i]),
        selected_without_future_labels="first primary query in the source query order",
        confidence_status="Raw forecast; empirical calibration is reported separately, without a per-query accuracy guarantee.")
    write_json(OUT / "forecast_example.json", item)
    print(json.dumps(item), flush=True)


if __name__ == "__main__":
    example()

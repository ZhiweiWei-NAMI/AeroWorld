"""Export predictions of a trained compact predictor (model.pt) for a cached cell set.

``--cells dev`` rebuilds the held-out TRAIN-profile selection set exactly as in
train_compact (dev episodes, cutoff stride 5); other names load ``CACHE/NAME.npz``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Dataset.world_model.model.compact_model import (CompactStatePredictor, batch_tensors, cell_units, law_terms,
                                                     prediction_rows)
from Dataset.world_model.model.train_compact import cached_cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cells", default="dev")
    args = parser.parse_args()
    checkpoint = torch.load(args.run/"model.pt", map_location="cpu", weights_only=False)
    options = checkpoint["args"]
    if args.cells == "dev":
        cells, _ = cached_cells(args.cache, "train_all", None)
        dev = options["dev_episodes"].split(",")
        keep = np.isin(cells["episode"], dev) & (cells["cutoff"] % 5 == 0)
        cells = {k: v[keep] for k, v in cells.items()}
    else:
        cells, _ = cached_cells(args.cache, args.cells, None)
    names = checkpoint["names"]
    dropped = [i for i, n in enumerate(names) if any(n.startswith(p) for p in options["drop_features"].split(",") if p)]
    if dropped:
        cells["X"] = cells["X"].copy(); cells["X"][:, dropped] = checkpoint["mean"][dropped]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactStatePredictor(len(names), len(checkpoint["vocab"]), int(options["hidden"]), int(options["depth"]),
                                  float(options["dropout"]), int(options["mixture"]), float(options.get("micro_scale", 0)) > 0).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    units = cell_units(cells, checkpoint["rates"])
    floor = float(options["scale_floor"])
    outputs = []
    with torch.no_grad():
        for start in range(0, len(cells["horizon"]), 65536):
            rows = np.arange(start, min(start+65536, len(cells["horizon"])))
            batch = batch_tensors(cells, rows, units, checkpoint["mean"], checkpoint["std"], device)
            raw = model(batch["x"], batch["category"], batch["activity"], batch["horizon"])
            outputs.append([t.cpu() for t in law_terms(raw, floor, batch["unit"], float(options.get("micro_scale", 0)))])
    terms = [torch.cat([o[i] for o in outputs]) for i in range(5)]
    with open(args.run/f"predictions_{args.cells}.jsonl", "w") as f:
        for start in range(0, len(terms[0]), 20000):
            rows = np.arange(start, min(start+20000, len(terms[0])))
            for row in prediction_rows(cells, rows, [t[rows] for t in terms], units, args.run.name):
                f.write(json.dumps(row)+"\n")
    print(json.dumps({"run": str(args.run), "cells": args.cells, "rows": len(terms[0])}))


if __name__ == "__main__":
    main()

# B01 schema and causal data specification

The recovered B01 body contained S01.01–S01.04 in full. This current version retains those four responsibilities and replaces obsolete paths and process instructions with the executable interfaces. It does not reconstruct missing later stages. Production inputs remain read-only; current numerical results are in [README.md](README.md).

## S01.01 — Canonical record schema

`Dataset/world_model/p01_schema_rollout_v4/src/p01v4/contracts/schema.py` declares static field families, entity/observer identities, states, relations, events and observation indices. `contracts/schema-v2.json` is the serialized declaration. Boolean, enumerated, real, bounded, circular, vector, optional and ragged fields carry applicable units, frames and legal values.

A recorded zero, an absent entity, an inapplicable field and an unavailable value have different semantics. The schema rejects undeclared field families, malformed frames, non-finite numbers and illegal values at their field path. New semantic field families require a declaration and authentic source mapping; free strings are not silently converted to finite prediction classes. Dynamic query compilation reports unsupported open-string families before inspecting future labels.

The numerical pilot uses the declared pose/motion fields and real communication fields. Distribution laws follow the declaration. The schema permits more types than this one window exercises; all evaluated future labels in the current window are present, so absent/inapplicable transitions have no empirical score yet.

## S01.02 — Authentic episode import

`data/import_episode.py` imports truth frames, roster, graph deltas, events and observation paths, retaining source references and archive-only source facets. The current local product is `derived/pilot_L4-1_v1__seed00.canonical-post-v11.jsonl`, with source inventory in `manifests/pilot-import-post-v11.json`. The canonical import contains 180,413 records. Its selected RGB/LiDAR observation indices are historical import coverage, not a statement that the full episode has only those modalities.

`data/observations.py` independently materializes the actual capture archive's RGB, depth, class segmentation and LiDAR, including calibration, times, source members and hit masks. `derived/pilot_observations/inventory.json` owns full capture coverage. The actual archive has 180 captures per modality at ticks 5–900; missing tick 0 is explicit. Six selected frames are materialized for geometry, while the model uses one historical capture at tick 5.

`model/causal_input.py` adds the authentic communication producer and the initial world-truth graph when building the model input. The renderer's hardcoded latency/loss/status annotations remain in archival evidence but are excluded from model-visible text and supervision. A source location is the traceability mechanism; no digest or guessed replacement is used.

## S01.03 — Reversible structured input

`data/serialization.py` supplies JSONL, compact-table and typed dictionary views under `configs/serialization-v2.json`. Decoding preserves parsed source values exactly, including list order, frame tags and explicit non-present kinds. Reserved wire shapes are escaped, so a present string or object cannot be mistaken for a missing-value marker. Archive and model-visible views remain separate.

`model/causal_input.py::semantic_text` builds the actual Qwen text: static schema, identities and sensor definitions followed by frozen-prefix symbol/change records. Its exact roundtrip covers retained model-visible causal semantics. Provenance bookkeeping, future scripts and renderer defaults are outside that projection. Source text and bindings are saved with the result.

Token counts come from the local Qwen tokenizer, not byte counts or entity counts. The current structured prefix has 219,777 tokens; native RGB and sensor insertion have additional tokens. The prefix-wide alias table currently changes when new real history is encoded, so arbitrary real-history cache continuation is not implemented. Predicted appends retain the already fixed prefix.

PyG joins use episode, tick and raw entity identity, not a claim that canonical record ID equals a PyG node index. Typed projection dictionaries preserve node, edge, type and state information for graph consumers; the executed backbone route uses their causal text representation.

## S01.04 — Targets, cutoff and split scope

`data/windows.py`, `normalization.py` and `provenance.py` enforce explicit input cutoff, availability time and original-source split membership. `configs/split-table.json` retains the established scenario/seed assignments. Copies and decorated variants share the original split group. The listed historical cohort cutoffs describe that cohort; they do not reassign the original when the current pilot uses cutoff 5.

The current numerical window is recorded in `configs/pilot-windows-v2.json`: TRAIN history 0–5, target 10 and second/direct target 15. Input normalization and temporal scales use only that TRAIN prefix. The future encoder is a fixed target branch, and future state/visibility never selects the input entity or query set. Future true poses, observations and states are not appended during rollout.

Code indexes the canonical and communication inputs once and reuses typed data and immutable observation arrays. It does not repeat provenance scans or fit data distributions inside each numerical query. The experiment does not use VALID/TEST for fitting. Results on neighboring times of this TRAIN episode are reported as a local closed loop, not as held-out prediction or calibration.

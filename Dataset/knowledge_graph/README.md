# World ontology

`world_core.ttl` defines shared world concepts. `domain/` and `events/` contain the current domain and event definitions, with their source definitions in each `catalog.yaml`. `profiles/ontology_selection_manifest.json` declares the selected modules.

Executable objective rules live in `../semantic_rules/`; ontology definitions alone do not establish that an event occurred. Runtime state, source evidence and computed predicates are required by `../semantic_truth/objective_pipeline.py`.

# Objective truth

This directory derives typed world state, predicates, events and graph records from recorded episode inputs. `objective_pipeline.py` is the producer entrypoint; `episode_sources.py` and `input_adapter.py` bind its source records. `provenance.py` retains source references while omitting integrity metadata from published artifacts.

The contracts live in `../semantic_rules/`. Execution models live in `../semantic_simulation/`. Raw episodes and generated truth remain in external `aw_data/` storage; this source checkout does not include them. A successful import does not establish that an episode has been regenerated or accepted.

# Recorded execution and state models

This directory computes model state from recorded motion, weather, task and action inputs. It also contains the native packet transport integration. Model estimates, actual packet receipts, authored parameters and sensor observations retain distinct meanings.

The current P09 entrypoints are [L6 receipt execution](p09_l6_5_receipt_v1/README.md), [L2 objective input repair](p09_dimension_repair_v1/README.md), and `p09_core_sources.py` for source-bound lifetime, energy and compute products. Run the latter with explicit `--project-root`, `--runtime-root` and `--output` paths. Its source index records applicability and whether P01 consumes each product; creating a side table does not update an existing training graph.

Episode data, runtime binaries and result packages remain outside Git. `../tools/p09_release.py` assembles UE inputs from completed recorded runs and does not execute UE capture.

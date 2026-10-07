# L6-5 command receipts

This module runs the original `L6-5_v1__seed00` action sequence through actual ns-3 packet reception and the event interpreter. The source scene, script, radio configuration, action grants and motion speeds are retained.

The current server result is `/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/l6_receipt_r2`. Three transport/trajectory iterations converged. All nine native packets were received. The abnormal action executes at tick 275 at its authored 9 m/s; lockout executes at 310 at 4 m/s. GCS lockout takes effect at 311, UAV lockout at 315, GCS secure state at 386, UAV recovery at 390. Landing reception becomes visible at 466; the five-tick event grid dispatches at 470. The UAV reaches landed at 588, has zero velocity at 589, and supplies terminal capture-grid evidence at 590.

Recovery does not reactivate the revoked abnormal-movement authority. Landing retains its existing action-specific grant.

The authored landing feasibility estimate is retained under `authored_terminal_estimate` with its script reference. It is not evidence for the actual dispatch time or terminal outcome.

## Reproduction

Run from the checkout root with the project's airfogsim Python and its existing ns-3 runtime. Data and provider binaries remain outside Git. Supply the immutable R1 and published trajectories explicitly when they are stored outside this checkout:

```bash
/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python -B -m Dataset.semantic_simulation.p09_l6_5_receipt_v1.real_prefix_run \
  --output-dir /mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/l6_reproduction \
  --r1-trajectory /mnt/data2/weizhiwei/AERO_WORLD/design/p09/mechanism_completion_v1/l6_5_command_receipt_revision/real_prefix_run_r1/trajectories.jsonl \
  --published-trajectory /mnt/data2/weizhiwei/AERO_WORLD/aw_data/render_ready_episodes_capture_filtered/L6-5_v1__seed00/trajectories.jsonl
```

Choose a new output directory. The code refuses completed results. The external reference paths above must exist; no substitute trajectories are generated. The scenario and radio inputs shipped with this source are `Dataset/scenarios/L6_digital_layer/failure/L6-5_v1/` and `design/p01_ns3/radio_reference_R1.json`. `provider_adapter.py` defines the native runtime requirements.

`adapter.py` owns command identity and receipt admission; `single_action_prefix.py` connects admission to actual interpreter/handler execution; `real_prefix_run.py` iterates native transport and trajectories and writes the result. Source-only import checks do not establish a successful simulation.

## Data and capture impact

Relative to the published dense trajectory, the first UAV position difference is 271; on the capture grid it is 275. The capture comparison has 44 position-changing rows plus three velocity-only quantization differences at 75/80/85 (maximum about 8e-6 m/s). Relative to R1, the first position difference is 466. GCS positions are unchanged.

The converter also recovers the scene-declared yaw of 35 degrees for the initial vertical flight: the first changed capture orientation is tick 35, where the old publication used an unsupported zero. This source repair changes orientation, while the R2 motion comparison above describes position. The current result changes no weather values, actor appearance or sensor configuration. The user performs UE capture from the assembled current package; this server run produced no RGB, depth, segmentation or LiDAR captures. The full episode retains ticks 0..900; capture scheduling remains every 5 ticks.

## Objective projection

`objective_projection.py` assembles the two actual R2 target streams, each covering ticks 0..900, with recorded background trajectories. Observer visibility remains a separate input. It uses the current scene through the existing explicit static-geometry input and runs the shared objective producer; actions and event traces remain execution evidence, not preassigned event labels.

```bash
/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python -B -m Dataset.semantic_simulation.p09_l6_5_receipt_v1.objective_projection \
  --project-root /mnt/data2/weizhiwei/AERO_WORLD \
  --receipt-dir /mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/l6_receipt_r2 \
  --capture-dir /mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/ue_input_current/capture_filtered_updates/L6-5_v1__seed00 \
  --output-dir /mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/l6_objective_reproduction
```

A completed run writes `current_binding.json` with its actual closure, source paths and generated files. A rejected objective graph does not count as a completed run. The runtime directory and existing external inputs must be available; this command does not rerun native transport or UE capture.

The current objective result is `/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/l6_objective_current`. Its actual run took 467.226 seconds, wrote 33 artifacts, and rejected no graph records. It contains 9 events, 9 event outcomes, 52 predicate transitions, 312 continuity records, 163 semantic graph deltas and 180 world graph deltas. Its assembled inputs match the current capture package, including the corrected background UAV yaw.

Closure is `PASS + EXPLICIT_GAPS`: the GCS compromise stage passes at tick 245; the other three stages remain explicitly not applicable under the current contract. The recorded position never exceeds the existing 15 m planned-route deviation threshold. Native packet admission, the runtime unauthorized-command flag, and authentication rejection have different meanings. The current ontology still lacks a typed command/control producer for these receipts; successful lockout does not establish security-control unavailability. Runtime secure recovery at 390 also does not establish recovery from geometric route deviation, which did not occur. The complete native receipts and executed state changes remain available as evidence without inventing these ontology positives.

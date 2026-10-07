# L6-5 command receipts

This module executes the original L6-5 action sequence through native ns-3 reception and the event interpreter. It retains the authored scene, radio configuration, action grants and motion speeds.

Current raw results live in `Dataset/episodes/L6-5_v1__seed00/`. `actual_execution_trajectories.jsonl` contains the native target trajectories; `trajectories.jsonl` contains the assembled world. Receipts, executed actions, physical evidence and terminal states remain separate from typed ontology events.

Run from the project root with the airfogsim Python and the existing ns-3 runtime:

```bash
/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python -B -m Dataset.semantic_simulation.p09_l6_5_receipt_v1.real_prefix_run
/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python -B -m Dataset.semantic_simulation.p09_l6_5_receipt_v1.objective_projection
```

These commands regenerate the current raw and objective outputs. The native runner accepts an explicit `--output-dir`; the projection accepts `--project-root`, `--receipt-dir`, `--capture-dir` and `--output-dir`. The projection reads the existing capture-filtered world, combines recorded background motion with the actual target streams, and calls the shared semantic producer. Its default output is `aw_data/objective_semantic_truth/L6-5_v1__seed00/`; `current_binding.json` records source paths and the actual closure.

`adapter.py` owns command identity and receipt admission. `single_action_prefix.py` connects receipt admission to interpreter execution. `real_prefix_run.py` iterates native transport and trajectories. The source scene and script are under `Dataset/scenarios/L6_digital_layer/failure/L6-5_v1/`; the radio reference is `design/p01_ns3/radio_reference_R1.json`. `provider_adapter.py` defines runtime requirements.

Recovery does not reactivate revoked abnormal-movement authority. The landing action retains its own grant. Authored feasibility estimates remain distinct from actual dispatch and terminal evidence. The recorded native lockout and recovery do not establish ontology predicates for geometric route deviation or security-control unavailability. Consult the actual closure and event outputs for their defined meanings and gaps.

This module produces simulation truth and UE inputs. It does not capture RGB, depth or segmentation.

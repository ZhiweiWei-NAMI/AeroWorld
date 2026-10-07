# L2 objective inputs

This module joins the adopted `L2-1_v2__seed00` execution records to the shared objective truth pipeline. `runtime_merge.py` preserves typed runtime state; `objective_rebuild.py` stages the exact declared sources and invokes the current producer.

The accepted recorded result is loss PASS at tick 265 and recovery FAIL. The adopted execution explicitly omits the tower recovery patches at 300 and 395. UAV movement and backup-link use do not establish tower recovery. The original published episode instead has tower recovery at 295; these are distinct executions.

The completed formal-objective run bound 901 recorded dust values from the original normalized weather file and restored five link-loss transitions. The saved builder time is 368.673275 s and shell wall time 370.064 s. This wiring changes no poses or weather values and does not require UE recapture. The inherited dust = 0 values are historical converter output, not measured atmospheric dust; retain that origin qualification when consuming the accepted result.

The current evidence remains outside Git at `design/p09/mechanism_completion_v1/numeric_mapping_checkpoint/builder_short_session_v1/author_formal_objective_checkpoint_v8/` in the original working tree. `formal_objective_receipt_r2.json`, `link_loss_diagnosis.json` and `objective_r2/L2-1_v2__seed00/epi_closure.json` distinguish the completed run from earlier attempts. The source also requires the original episode under `aw_data/render_ready_episodes/L2-1_v2__seed00` and the explicitly named adopted records under `/mnt/data1/weizhiwei/AERO_WORLD_runtime/p09/`; inspect its path constants before reproducing. This source-only checkout does not contain those data products.

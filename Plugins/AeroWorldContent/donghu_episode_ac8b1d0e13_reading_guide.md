# Donghu Episode `episode_ac8b1d0e13` Reading Guide

## Purpose

This document records the current state of the running Donghu episode `episode_ac8b1d0e13`, explains which exported artifacts should be read for UE integration, and documents the key fields needed for rendering, data collection, and later optimization work under `E:\Dynamic City`.

The goal is to make one thing explicit:

- The authoritative source for per-tick world truth is `truth_frames.jsonl`.
- UE-facing rendering should consume only the minimum state needed for the current tick.
- ROI rendering should be treated as a view/window over one global truth timeline, not as separate simulated worlds.

## Snapshot At Documentation Time

Inspection snapshot was taken on `2026-04-06` from the currently growing episode directory:

- Episode directory: `artifacts/episodes/episode_ac8b1d0e13`
- Episode ID: `episode_ac8b1d0e13`
- Map ID: `donghu_road_topo`
- Render mode currently recorded in truth frames: `a_only`
- Active ROI: `roi.intersection_a.v1`
- Frames written so far at inspection time: `671`
- Tick range written so far: `0..670`
- Sim time reached at inspection time: `67.0 s`
- Tick continuity so far: contiguous, no missing ticks observed

This episode is still running. The frame count and max tick will continue to grow until the session exits and flushes its final exports.

## Current Global Roster

The current Donghu roster for this episode is stable across ticks. Frame `0` reports:

- Global entity count: `53`
- Intersection A visible roster: `49`
- Offstage retained entities from Intersection B: `4`

Category counts:

- Vehicles: `25`
- Pedestrians: `13`
- UAVs: `2`
- Facilities: `10`
- Other: `3`

Site counts:

- `site.intersection_a`: `49`
- `site.intersection_b`: `4`

Signal IDs present in the global roster:

- `signal_a_ns`
- `signal_a_ew`
- `signal_b_main`

Interpretation:

- The simulation is one global world.
- ROI A is the current render window.
- ROI B entities are still preserved in the same tick, but can be marked `offstage` and withheld from UE submission.

## Artifact Inventory

The current episode directory contains these major files:

- `truth_frames.jsonl`
  Purpose: authoritative per-tick world truth for all entities in the global roster.
  Use this for UE frame playback, synchronized multi-camera capture, and exact tick-to-tick replay.

- `ue_resolved_frames.jsonl`
  Purpose: UE-side resolved or grounded state returned after scene sync.
  Use this for truth-vs-UE comparison, not as the primary render input.

- `runtime_frames.jsonl`
  Purpose: full runtime snapshot including state stores, planner, providers, and metadata.
  Use this for debugging or inspection, not for the render path.

- `trajectories.jsonl`
  Purpose: one row per entity per tick with flattened trajectory information.
  Use this for lightweight analytics, time-series extraction, entity-centric queries, and dataset preprocessing.

- `scheduler.jsonl`
  Purpose: planner/task/procedure summary per tick.
  Use this only if you need dispatch, task, or procedure state.

- `event_trace.jsonl`
  Purpose: event chain and semantic event timeline.
  Use this to label incidents, weather changes, dispatch windows, and recovery windows.

- `facility_sessions.jsonl`
  Purpose: resource allocations and facility session state.
  Use this only for infrastructure/resource behavior analysis.

- `comm.jsonl`
  Purpose: communication/network metrics by entity.
  Use this if communication quality needs to affect rendering or labels.

- `weather_meta.jsonl`
  Purpose: weather metadata per tick.
  Use this to annotate captures or drive environment presets.

- `global_entity_roster.json`
  Purpose: static roster baseline.
  Read this once at startup to know IDs, categories, sites, and base positions.

- `scenario_plan.json`
  Purpose: static plan summary, ROI windows, seeds, and site contracts.
  Read this once at startup to configure capture, ROI logic, and expectations.

- `logs.jsonl`
  Purpose: verbose execution log.
  Do not put this in any render or live data path.

## Which File Should Be Read For UE

For UE visualization, the preferred priority is:

1. `truth_frames.jsonl`
2. `global_entity_roster.json`
3. `scenario_plan.json`

Recommended division:

- `truth_frames.jsonl` provides dynamic state for the current tick.
- `global_entity_roster.json` provides stable metadata and can be cached once.
- `scenario_plan.json` provides ROI windows and site contracts and can also be cached once.

Avoid using `runtime_frames.jsonl` in the UE render loop. It is larger, more nested, and contains far more than the renderer needs.

## What Each Tick Already Contains

Each line in `truth_frames.jsonl` is one complete tick/frame.

Top-level frame keys currently used:

- `episode_id`
- `frame_id`
- `tick`
- `frame_seq`
- `sim_time_s`
- `dt_s`
- `tick_hz`
- `map_id`
- `render_mode`
- `active_roi_id`
- `active_roi_bounds_enu_m`
- `entities`
- `roster_summary`
- `coordinate_contract`
- `authority_contract`
- `contract_assertions`

This is enough to guarantee:

- deterministic tick identity
- exact simulation time
- consistent ENU coordinates
- one authoritative world state per tick

## Entity Payload Shape In `truth_frames.jsonl`

Each entity currently has this stable top-level structure:

- `entity_id`
- `entity_kind`
- `entity_category`
- `site_id`
- `proxy_template_id`
- `truth_pose`
- `render_presence`
- `annotations`
- `tags`
- `state_revision`
- `visual_revision`

This is already suitable for UE playback.

### `truth_pose`

The authoritative motion payload lives here.

Current keys:

- `position_enu_m`
- `rotation_deg`
- `velocity_enu_mps`
- `coordinate_contract_id`
- `authority_owner`
- `authority_mode`

Meaning:

- `position_enu_m`: exact entity position in ENU metres
- `rotation_deg`: current orientation
- `velocity_enu_mps`: velocity vector in ENU metres per second
- `coordinate_contract_id`: coordinate convention identifier
- `authority_owner`: who owns truth pose
- `authority_mode`: authoritative source mode

For UE rendering, the minimum required subset is:

- `entity_id`
- `truth_pose.position_enu_m`
- `truth_pose.rotation_deg`
- `truth_pose.velocity_enu_mps`

### `render_presence`

This determines whether the entity should be submitted or withheld for the current ROI/view.

Current keys:

- `global_roster`
- `roi_membership`
- `submission_state`
- `visibility_state`
- `offstage`
- `offstage_reason`

Meaning:

- `submission_state == "submit_to_ue"`: send actor update/spawn for this tick
- `submission_state == "retain_offstage"`: keep it in the global truth timeline but do not render it in the current ROI pass
- `visibility_state`: `visible`, `hidden`, or `offstage`

For UE memory and draw-call control, this is the field group that matters most.

### `annotations`

This is category-dependent state.

Observed examples:

- Vehicle annotations:
  - `lane_id`
  - `approach_id`
  - `signal_phase`
  - `queue_length`
  - `density`
  - `speed_mps`
  - `weather`
  - `state_facets.network`

- Pedestrian annotations:
  - `activity_type`
  - `path_id`
  - `speed_mps`
  - `weather`
  - `state_facets.activity`

- UAV annotations:
  - `orbit_radius_m`
  - `speed_mps`
  - `weather`
  - `state_facets.activity`
  - `state_facets.assignment`
  - `state_facets.health`
  - `state_facets.local_environment`
  - `state_facets.network`
  - `state_facets.operational`
  - `state_facets.resource_sessions`

- Signal annotations:
  - `signal`
  - `weather`
  - `state_facets.activity`
  - `state_facets.signal_control`

## Minimum UE Fields To Consume

If the purpose is only visualization and data collection, the minimal render payload per entity per tick should be flattened to:

- `entity_id`
- `entity_kind`
- `entity_category`
- `site_id`
- `position_enu_m`
- `rotation_deg`
- `velocity_enu_mps`
- `speed_mps`
- `action`
- `posture`
- `visibility_state`
- `submission_state`
- `offstage`
- `status`

Recommended derivation rules:

- `speed_mps`
  Use `annotations.speed_mps` when present.
  Otherwise derive from `truth_pose.velocity_enu_mps`.

- `action`
  For pedestrians, prefer `annotations.state_facets.activity.activity_type`.
  For signals, use `signaling`.
  For vehicles, derive `moving` or `idle` from speed if no explicit activity exists.
  For UAVs, derive from activity if present, else from speed.

- `posture`
  For pedestrians, use `annotations.state_facets.activity.posture`.
  For UAVs, `hover` is a useful default.
  For signals/facilities, `fixed` is acceptable.

- `status`
  Flatten selected state values only, such as:
  - `health`
  - `operational_mode`
  - `mission_status`
  - `signal_phase`
  - `weather`

This is the right shape for `E:\Dynamic City`.

## Why `assignment` Is Not A UE Render Requirement

`assignment` exists to support planner/procedure/runtime compatibility for UAV tasking.
It is not required for basic rendering.

You only need it when:

- you want to visualize current mission targets
- you want to color-code dispatch intent
- you want to overlay resource usage or route intent

For pure visualization and dataset generation, `assignment` can be ignored in the UE consumer.

Recommendation:

- Keep `assignment` in the backend truth exports.
- Do not require the UE side to parse it for baseline rendering.

## Recommended Read Path For Different Tasks

### 1. Real-time or near-real-time UE playback

Read:

- `truth_frames.jsonl` tail/stream only
- cache `global_entity_roster.json` once
- cache `scenario_plan.json` once

Do not read:

- `runtime_frames.jsonl`
- `logs.jsonl`
- `scheduler.jsonl`

### 2. Dataset generation and camera metadata binding

Read:

- `truth_frames.jsonl`
- `event_trace.jsonl`
- `weather_meta.jsonl`

This gives:

- synchronized world truth
- event labels
- weather labels

### 3. Entity trajectory analytics

Read:

- `trajectories.jsonl`

This file is much easier for:

- querying one entity across many ticks
- building CSV/parquet tables
- plotting speed, path, and state over time

### 4. Planner/procedure debugging

Read:

- `scheduler.jsonl`
- `runtime_frames.jsonl`
- `facility_sessions.jsonl`

## Necessary Read Optimizations

### A. Do not load the full `truth_frames.jsonl` into memory in the live path

While the episode is running, the file keeps growing. For UE playback, only the current tick or a small sliding window is needed.

Do:

- tail the latest line for live monitoring
- stream sequentially for replay
- build an index after the episode completes if random tick access is needed

Do not:

- repeatedly re-read the whole file for every tick

### B. Cache static files once

Read once at process startup:

- `global_entity_roster.json`
- `scenario_plan.json`

These files do not change during the episode.

### C. Use `render_presence.submission_state` as the first filter

Before actor update/spawn:

- if `submission_state == "submit_to_ue"`, render it
- if `submission_state != "submit_to_ue"`, skip it for the current ROI pass

This is the cheapest and cleanest way to control ROI-specific UE load.

### D. Separate frame consumption from camera capture

Recommended order per tick:

1. Read one truth frame
2. Apply entity transforms/state once in UE
3. Trigger all cameras for that same tick
4. Save outputs with `episode_id + frame_id + tick + camera_id`

Do not re-apply the same tick separately for each camera.

### E. Build a post-run tick index for random access

After the episode completes, build a sidecar index:

- key: `tick`
- value: byte offset in `truth_frames.jsonl`

This allows direct seek-to-tick access without scanning all prior lines.

Recommended sidecar file names:

- `truth_frames.tick_index.json`
- or `truth_frames.tick_index.msgpack`

### F. Flatten to a UE-facing JSONL once

The current canonical file is correct but nested.
For `E:\Dynamic City`, generate a simplified file after the episode stops:

- one line per tick
- one `entities` list
- only UE-relevant fields

Recommended file name:

- `ue_truth_frames.jsonl`

Recommended entity payload:

```json
{
  "entity_id": "drone_a1",
  "entity_kind": "uav.drone",
  "entity_category": "uav",
  "site_id": "site.intersection_a",
  "position_enu_m": [42.6, 30.0, 21.7],
  "rotation_deg": {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 180.0},
  "velocity_enu_mps": [-9.95, 0.0, -0.90],
  "speed_mps": 9.999998,
  "action": "idle",
  "posture": "hover",
  "visibility_state": "visible",
  "submission_state": "submit_to_ue",
  "offstage": false,
  "status": {
    "health": "healthy",
    "operational_mode": "enabled",
    "weather": "rain"
  }
}
```

## Useful Commands

### PowerShell: check the latest tick during runtime

```powershell
Get-Content artifacts\episodes\episode_ac8b1d0e13\truth_frames.jsonl -Tail 1
```

### PowerShell: inspect exported files

```powershell
Get-ChildItem artifacts\episodes\episode_ac8b1d0e13 | Select-Object Name,Length,LastWriteTime
```

### Python: stream frames without loading all rows

```python
import json
from pathlib import Path

path = Path(r"artifacts/episodes/episode_ac8b1d0e13/truth_frames.jsonl")
with path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        frame = json.loads(line)
        tick = frame["tick"]
        for entity in frame["entities"]:
            if entity["render_presence"]["submission_state"] != "submit_to_ue":
                continue
            position = entity["truth_pose"]["position_enu_m"]
            velocity = entity["truth_pose"].get("velocity_enu_mps", [0.0, 0.0, 0.0])
```

### Python: flatten to a lighter UE payload

```python
import json
from pathlib import Path

src = Path(r"artifacts/episodes/episode_ac8b1d0e13/truth_frames.jsonl")
dst = Path(r"artifacts/episodes/episode_ac8b1d0e13/ue_truth_frames.jsonl")

with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="\n") as fout:
    for line in fin:
        if not line.strip():
            continue
        frame = json.loads(line)
        entities = []
        for entity in frame["entities"]:
            truth_pose = entity.get("truth_pose", {})
            render_presence = entity.get("render_presence", {})
            annotations = entity.get("annotations", {})
            activity = annotations.get("state_facets", {}).get("activity", {})
            entities.append({
                "entity_id": entity["entity_id"],
                "entity_kind": entity.get("entity_kind", ""),
                "entity_category": entity.get("entity_category", "other"),
                "position_enu_m": truth_pose.get("position_enu_m", [0.0, 0.0, 0.0]),
                "rotation_deg": truth_pose.get("rotation_deg", {}),
                "velocity_enu_mps": truth_pose.get("velocity_enu_mps", [0.0, 0.0, 0.0]),
                "speed_mps": annotations.get("speed_mps", 0.0),
                "action": activity.get("activity_type", ""),
                "posture": activity.get("posture", ""),
                "submission_state": render_presence.get("submission_state", "submit_to_ue"),
                "visibility_state": render_presence.get("visibility_state", "visible"),
            })
        fout.write(json.dumps({
            "episode_id": frame["episode_id"],
            "frame_id": frame["frame_id"],
            "tick": frame["tick"],
            "sim_time_s": frame["sim_time_s"],
            "entities": entities,
        }, ensure_ascii=False) + "\n")
```

## Recommended UE Integration Model

Use one global truth timeline and multiple ROI/camera render passes.

Recommended approach:

- Keep one authoritative backend timeline for all entities.
- Use `render_presence` to decide which entities are submitted in the current ROI pass.
- For the same `tick`, allow multiple camera captures.
- Keep data aligned using:
  - `episode_id`
  - `frame_id`
  - `tick`
  - `camera_id`

Do not split the simulation into separate per-ROI worlds unless absolutely necessary.

This preserves:

- one ground-truth timeline
- exact cross-camera synchronization
- easier later labeling
- lower system complexity

## Practical Recommendation For `E:\Dynamic City`

For the next integration step, the cleanest pipeline is:

1. Treat `truth_frames.jsonl` as canonical backend truth.
2. Build or consume a flattened `ue_truth_frames.jsonl`.
3. Cache `global_entity_roster.json` and `scenario_plan.json` once.
4. In UE, update all actors once per tick.
5. In the same tick, trigger all cameras.
6. Persist outputs by `episode_id/frame_id/tick/camera_id`.

If the target is only visual rendering plus data capture, the UE consumer should ignore:

- planner internals
- procedure internals
- assignment internals
- facility session internals

unless a later visualization pass explicitly needs them.

## Bottom Line

The current running episode already exports the fields needed for UE rendering:

- per-tick entity position
- per-tick velocity
- per-tick render visibility/offstage status
- category-specific state such as speed, pedestrian action/posture, UAV state, and signal phase

The main optimization requirement is not changing the simulation output.
It is reading the right files with the right granularity:

- `truth_frames.jsonl` for synchronized frame playback
- `trajectories.jsonl` for lightweight analytics
- static caches for roster and scenario plan
- optional flattened UE JSONL for the actual consumer

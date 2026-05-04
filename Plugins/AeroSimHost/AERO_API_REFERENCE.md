# AeroSimHost External API Reference

> **Generated from source code** — authoritative contract for Python scripts controlling the UE simulation via AirSim RPC.

---

## 1. Transport & Envelope

All `simAero*` operations use the AirSim RPC channel (`msgpackrpc` over TCP, default port `41451`).

### Request Envelope

```json
{
  "api_version": "1.0",
  "request_id": "<uuid>",
  "map_id": "<map_id>",
  "payload": { ... }
}
```

- `request_id` — client-generated unique ID; echoed in response.
- `map_id` — optional; if provided and differs from the currently loaded map, the bridge auto-loads context for the new map.
- `payload` — operation-specific fields (documented per operation below).

### Response Envelope

```json
{
  "api_version": "1.0",
  "map_id": "<map_id>",
  "request_id": "<request_id>",
  "op": "<operation_name>",
  "status": "ok" | "error",
  "payload": { ... },
  "error": { "message": "..." }
}
```

- `status` = `"ok"` on success; `"error"` on failure.
- `error` object is present only when `status` = `"error"`.

### Position Conventions

- **Runtime control standard = ENU metres** — `position_enu_m`, `target_enu_m`, `pose_enu_m`, `point_enu_m` are the authoritative external-control coordinates.
- **UE internal space = world centimetres** — `position_world_cm`, `target_world_cm`, `pose_world_cm` are accepted by some APIs, but they are secondary to ENU for external Python control.
- Position fields accept **arrays** `[x, y, z]` or **objects** `{"x":..., "y":..., "z":...}` (also `{"east_m":..., "north_m":..., "up_m":...}` where supported).

### Coordinate Authority, GeoJSON, and Z Handling

**Authoritative map frame source:** `Config/LowAltitude/Maps/<map_id>/map_context.json`

Example (`donghu_road_topo`):

```json
{
  "geo_reference": { "lat": 30.5609, "lon": 114.3627, "alt": 24.0 },
  "local_frame": "ENU",
  "world_origin_policy": "fixed",
  "world_origin_cm": [0.0, 0.0, 0.0]
}
```

- **`local_frame`** — current runtime frame; the bridge reports `ENU`.
- **`world_origin_cm`** — UE world origin corresponding to ENU `(0, 0, 0)`.
- **`geo_reference`** — geographic reference used by the content/map pipeline.

### GeoJSON Source Data vs Runtime Control Coordinates

- GeoJSON files under `Content/Maps/.../*.geojson` use **geographic coordinates** — typically `[lon, lat, alt]`.
- Example: `Content/Maps/donghu_road_topo/road/road.geojson` contains lines such as `[113.287102806, 23.12196133, 13.595577201]`.
- These coordinates are **source map data**, **not** the values you should send directly to `simAero*` RPCs.
- The runtime bridge expects **map-local ENU metres** after the import/build pipeline has transformed geographic data into the local simulation frame.
- Pedestrian semantic files already follow this rule: `ped_nav_semantic.source.json` uses `polyline_enu_m`, `center_enu_m`, and related ENU fields.

### ENU ↔ UE World Conversion

- `world_cm = world_origin_cm + enu_m * 100`
- `enu_m = (world_cm - world_origin_cm) / 100`

The Python client now exposes helpers:

```python
AeroSimClient.enu_m_to_world_cm(position_enu_m, world_origin_cm)
AeroSimClient.world_cm_to_enu_m(position_world_cm, world_origin_cm)
```

### Z-Axis / Grounding Rules

- **Ground-bound pedestrians**
  - Use `position_enu_m` / `target_enu_m` in Python.
  - With `snap_to_ground=true`, UE may modify **both** `X/Y` and `Z`:
    - `simAeroPedSpawn` / `simAeroPedReset` first try road-topology-assisted grounding, then ground projection.
    - `simAeroPedSetTarget` / `simAeroPedCommitCross` ground the target point before issuing the command.
  - Always treat the **response payload** (`position_enu_m`, `target_enu_m`, `ground_source`) as authoritative.

- **Ground-bound vehicles / props / facilities**
  - Prefer ENU input.
  - For assets with `ground_snap_policy = "project_down"`, the asset placement system primarily preserves requested `X/Y` and resolves **`Z`** from the world ground trace, then applies `default_z_offset_m`.
  - Use `simAeroSpawnAsset` / `simAeroMoveAsset` return values, or `simAeroApplyFrame` resolved spawn/update results, as the authoritative final pose.

- **Pedestrian semantic projection (`simAeroProjectGround`)**
  - Input is `point_enu_m` only.
  - Projection may snap the point onto semantic sidewalk/crossing edges or anchors within `max_snap_distance_m` and reconcile semantic Z with traced ground Z.
  - The semantic-vs-trace Z mismatch tolerance in code is currently **0.5 m** (`MaxSemanticGroundZMismatchM`).
  - Response returns `projected_enu_m`, `surface_normal_enu`, `anchor_id`.

- **UAV / aerial actors**
  - Use explicit ENU `Z` as mission altitude in the local frame.
  - Do **not** enable ground snap for aerial assets unless intentionally landing or projecting to a pad.

- **Trigger volumes / airspace / prisms**
  - Use explicit ENU Z extents via `min_z_m` / `max_z_m`, or `base_z_m` + `height_m`.

### Recommended External Python Workflow for Accurate Unified Coordinates

1. Call `simAeroLoadContext(map_id)`.
2. Read `payload.local_frame`, `payload.world_origin_cm`, `payload.geo_reference`.
3. Keep all runtime simulation state in **ENU metres**.
4. If source data comes from GeoJSON / lon-lat-alt, convert it to the map-local ENU frame **before** calling `simAero*`.
5. For pedestrians or other ground-bound actors, either:
   - call `simAeroProjectGround(point_enu_m=...)` first, or
   - submit the approximate ENU point and use the **response** (`position_enu_m`, `target_enu_m`, `ground_source`) as the final authoritative pose.
6. For scene-sync (`simAeroApplyFrame`), read back the resolved `position_enu_m` / `position_world_cm` returned in spawn/update results after UE applies grounding/offsets.
7. For downstream logic, feedback, and bookkeeping, trust **returned ENU coordinates from the server** instead of the raw request values whenever grounding or semantic snapping is involved.

### Python Client

```python
from aero_sim_client import AeroSimClient
client = AeroSimClient(host="127.0.0.1", port=41451)
```

Located at `Plugins/SumoImporter/Scripts/aero_sim_client.py`.

---

## 2. System Operations

### 2.1 `simAeroDescribeCapabilities`

List all supported operations, config kinds, pedestrian variants, and modes.

| Field | Type | Description |
|-------|------|-------------|
| *(no payload required)* | | |

**Response payload:**
- `operations` — `string[]` — list of all supported `simAero*` operation names.
- `config_kinds` — `string[]` — reloadable config types (`asset_catalog`, `weather_profiles`, `scenario_objects`, `ped_nav_semantic`, `map_context`).
- `ped_variants` — `string[]` — available variant IDs.
- `ped_modes` — `string[]` — available pedestrian modes.
- `ped_montage_tags` — `string[]` — available animation montage tags.
- `local_frame` — current runtime coordinate frame (`ENU`).
- `current_map_id` — currently loaded map.

**Python:** `client.describe_capabilities()`

---

### 2.2 `simAeroLoadContext`

Load map context, asset catalog, scenario objects, pedestrian navigation, and weather profiles for a given map.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `map_id` | string | **yes** | Map identifier (also accepted in envelope) |

**Response payload:**
- `map_id` — loaded map ID.
- `map_context_path` — resolved map context path.
- `scenario_objects_path` — resolved scenario object path.
- `ped_nav_bundle_path` — resolved pedestrian semantic bundle path.
- `local_frame` — current runtime coordinate frame.
- `world_origin_policy` — current origin policy.
- `ue_level_name` — UE level name declared in `map_context.json`.
- `world_origin_cm` — ENU origin expressed in UE world cm.
- `geo_reference` — geographic reference for upstream lon/lat/alt conversion.

**Python:** `client.load_context("my_map")`

---

### 2.3 `simAeroReloadConfig`

Hot-reload a specific configuration file without reloading the full map context.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kind` | string | **yes** | One of: `asset_catalog`, `weather_profiles`, `scenario_objects`, `ped_nav_semantic`, `map_context` |
| `path` | string | no | Override file path; defaults to the standard path for the kind |

**Response payload:**
- `kind` — echoed kind.
- `path` — resolved path that was loaded.

**Python:** `client.reload_config("asset_catalog")`

---

## 3. Scene Synchronisation (External Tick Model)

### 3.1 `simAeroApplyFrame`

**Core operation for external-tick-driven simulation.** Apply a batch of entity spawn, update, and remove deltas in a single frame. The frame context is recorded by the feedback subsystem so that collision/overlap events are tagged with the corresponding tick.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tick` | int | recommended | External simulation tick counter |
| `frame_id` | int | recommended | Monotonically increasing frame identifier |
| `sim_time_s` | float | no | Simulation time in seconds |
| `sample_seq` | int | no | Sample sequence number |
| `episode_id` | string | no | Episode identifier for grouping events |
| `spawns` | array | no | Entity spawn deltas (see below) |
| `updates` | array | no | Entity update deltas (see below) |
| `removes` | array | no | Entity remove deltas (see below) |

#### Spawn Delta Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entity_id` | string | **yes** | Unique entity identifier |
| `proxy_template_id` | string | **yes** | Must match a `logical_asset_id` in `asset_catalog.json` |
| `pose_enu_m` | object | **yes** | `{"position_enu_m": [x,y,z], "rotation_deg": {"yaw_deg":...}}` |
| `tags` | string[] | no | Query tags for this entity |
| `visual_state` | object | no | See §7 Visual State |

#### Update Delta Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entity_id` | string | **yes** | Must match a previously spawned entity |
| `pose_enu_m` | object | no | New pose (if omitted, keeps previous) |
| `tags` | string[] | no | Updated tags |
| `visual_state` | object | no | Updated visual state |

> If `entity_id` has not been spawned yet, an update delta with `proxy_template_id` will auto-promote to a spawn.

#### Remove Delta Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entity_id` | string | **yes** | Entity to remove |

**Response payload:**
- `tick`, `frame_id`, `sim_time_s`, `sample_seq`, `episode_id` — echoed.
- `spawns`, `updates` — arrays of objects containing:
  - `entity_id`
  - `proxy_id`
  - `logical_asset_id`
  - `position_enu_m`
  - `position_world_cm`
  - `actor_name` (if spawned)
  - `ground_source` (when ground resolution was applied)
- `removes` — arrays of `{"entity_id":..., "proxy_id":...}`.

**Python:** `client.apply_frame(payload, map_id="my_map")`

**Example (traffic_manager pattern):**

```python
client.apply_frame({
    "tick": 42,
    "frame_id": 100,
    "sim_time_s": 2.1,
    "episode_id": "ep_001",
    "spawns": [{
        "entity_id": "vehicle_001",
        "proxy_template_id": "vehicle.service.box.v1",
        "pose_enu_m": {
            "position_enu_m": [10.0, 20.0, 0.0],
            "rotation_deg": {"yaw_deg": 90.0}
        },
        "tags": ["vehicle", "traffic"]
    }],
    "updates": [{
        "entity_id": "vehicle_002",
        "pose_enu_m": {
            "position_enu_m": [15.0, 25.0, 0.0],
            "rotation_deg": {"yaw_deg": 45.0}
        }
    }],
    "removes": [{"entity_id": "vehicle_003"}]
}, map_id="my_map")
```

---

### 3.2 `simAeroPollFeedback`

Poll collision and overlap events accumulated since a given tick or frame.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `since_tick` | int | no | Return events with `tick > since_tick` |
| `since_frame_id` | int | no | Return events with `frame_id > since_frame_id` |

> If neither is provided, **all** buffered events are returned.

**Response payload:**
- `events` — array of feedback event objects (see §6).
- `upto_tick` — highest tick in the returned events.
- `upto_frame_id` — highest frame_id in the returned events.
- `episode_id` — current episode ID.

**Consumption pattern:** Save `upto_tick` and pass it as `since_tick` on the next poll to avoid re-reading events.

**Python:** `client.poll_feedback(since_tick=41)`

---

## 4. Pedestrian Operations

All pedestrian operations require the pedestrian to be managed by the `AeroRuntimeOrchestrationSubsystem`.

### 4.1 `simAeroPedSpawn`

Spawn a new pedestrian.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ped_id` | string | **yes** | Unique pedestrian identifier |
| `position_world_cm` / `position_enu_m` | vector | **yes** | Spawn position |
| `yaw_deg` | float | no | Initial facing direction (default 0) |
| `variant_id` | string | no | Appearance variant from the variant catalog |
| `snap_to_ground` | bool | no | Snap to ground surface (default `true`) |

**Response payload:** `ped_id`, `yaw_deg`, `position_world_cm`, `position_enu_m`, `used_provided_ground_point`, `ground_source`, `variant_id`.

**Python:** `client.ped_spawn("PED_001", position_enu_m=(x, y, z), yaw_deg=90, variant_id="adult_male_commuter")`

---

### 4.2 `simAeroPedReset`

Reset an existing pedestrian's position.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ped_id` | string | **yes** | |
| `position_world_cm` / `position_enu_m` | vector | **yes** | New position |
| `yaw_deg` | float | no | |
| `snap_to_ground` | bool | no | Default `true` |

**Response payload:** `ped_id`, `yaw_deg`, `position_world_cm`, `position_enu_m`, `used_provided_ground_point`, `ground_source`.

**Python:** `client.ped_reset("PED_001", position_enu_m=(x, y, z), yaw_deg=90)`

---

### 4.3 `simAeroPedSetTarget`

Set a walk target for a pedestrian.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ped_id` | string | **yes** | |
| `target_world_cm` / `target_enu_m` | vector | **yes** | Walk destination |
| `speed_cm_per_sec` | float | no | Walk speed; 0 = default |
| `snap_to_ground` | bool | no | Default `true` |

**Response payload:** `ped_id`, `speed_cm_per_sec`, `target_world_cm`, `target_enu_m`, `ground_source`.

**Python:** `client.ped_set_target("PED_001", target_enu_m=(x, y, z), speed_cm_per_sec=140)`

---

### 4.4 `simAeroPedObserve`

Play the "observe" (look-around) montage on a pedestrian.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ped_id` | string | **yes** | |
| `start_section` | string | no | Montage section to start from |

**Python:** `client.ped_observe("PED_001")`

---

### 4.5 `simAeroPedPlayAnimation`

Play an arbitrary animation montage on a pedestrian.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ped_id` | string | **yes** | |
| `animation_asset_path` | string | **yes** | UE asset path of the animation montage |
| `start_section` | string | no | Montage section to start from |
| `play_rate` | float | no | Playback speed multiplier (default 1.0) |
| `loop_count` | int | no | Number of loops (default 1, minimum 1) |

**Known animation montage paths** (from PROGRESS_NOTES):
- `/AeroSimHost/Animations/AM_Observe` — look-around
- `/AeroSimHost/Animations/AM_Wave` — waving
- `/AeroSimHost/Animations/AM_Phone` — phone usage
- `/AeroSimHost/Animations/AM_Crouch` — crouching
- `/AeroSimHost/Animations/AM_Point` — pointing
- `/AeroSimHost/Animations/AM_Sit` — sitting

**Python:** `client.ped_play_animation("PED_001", "/AeroSimHost/Animations/AM_Wave", play_rate=1.0, loop_count=2)`

---

### 4.6 `simAeroPedCommitCross`

Commit a pedestrian to cross (walk to target with crossing animation sequence).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ped_id` | string | **yes** | |
| `target_world_cm` / `target_enu_m` | vector | **yes** | |
| `speed_cm_per_sec` | float | no | |
| `snap_to_ground` | bool | no | Default `true` |

**Response payload:** `ped_id`, `speed_cm_per_sec`, `target_world_cm`, `target_enu_m`, `ground_source`.

**Python:** `client.ped_commit_cross("PED_001", target_enu_m=(x, y, z), speed_cm_per_sec=140)`

---

### 4.7 `simAeroPedStop`

Stop a pedestrian's current movement/animation.

| Field | Type | Required |
|-------|------|----------|
| `ped_id` | string | **yes** |

**Python:** `client.ped_stop("PED_001")`

---

### 4.8 `simAeroPedSetVariant`

Switch a pedestrian's visual appearance variant.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ped_id` | string | **yes** | |
| `variant_id` | string | **yes** | Variant ID from the variant catalog |

**Python:** `client.ped_set_variant("PED_001", "civilian_01")`

---

### 4.9 `simAeroPedRelease`

Release a pedestrian from the orchestration subsystem (despawn).

| Field | Type | Required |
|-------|------|----------|
| `ped_id` | string | **yes** |

**Python:** `client.ped_release("PED_001")`

---

### 4.10 `simAeroPedSpawnCrowd`

Spawn a group of pedestrians at a given origin.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `group_id` | string | no | Group identifier (default `"crowd.default"`) |
| `count` | int | no | Number of pedestrians to spawn |
| `seed` | int | no | Random seed |
| `spawn_origin_world_cm` / `spawn_origin_enu_m` | vector | no | Center of spawn area |
| `spawn_box_extent_cm` | vector | no | Half-extents of the spawn box |
| `yaw_policy` | string | no | `"random"` or `"fixed"` |
| `fixed_yaw_deg` | float | no | Used when `yaw_policy="fixed"` |
| `snap_to_ground` | bool | no | Default `true` |
| `appearance_pool_path` | string | no | UE asset path override |
| `role_profile_path` | string | no | UE asset path override |

**Response payload:** `group_id`, `skipped_count`, `seed`, `spawned_ids`.

**Python:** `client.ped_spawn_crowd("group_a", 10, spawn_origin_enu_m=(x, y, z), seed=42)`

---

### 4.11 `simAeroPedClearCrowd`

Remove all pedestrians in a crowd group.

| Field | Type | Required |
|-------|------|----------|
| `group_id` | string | **yes** |

**Python:** `client.ped_clear_crowd("group_a")`

---

### 4.12 `simAeroPedRespawnCrowd`

Respawn an existing crowd group with a new seed.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `group_id` | string | **yes** | |
| `seed` | int | no | New random seed |

**Python:** `client.ped_respawn_crowd("group_a", seed=123)`

---

## 5. Asset Operations

Assets are registered in `asset_catalog.json` as templates identified by `logical_asset_id`.

### 5.1 `simAeroSpawnAsset`

Spawn an asset instance from a catalog template.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `template_id` / `logical_asset_id` | string | **yes** | Must match a template in `asset_catalog.json` |
| `asset_id` | string | no | Instance ID (auto-generated if omitted) |
| `pose_enu_m` / `pose_world_cm` | object | either | Pose object including position + rotation |
| `position_enu_m` / `position_world_cm` | vector | either | Position shorthand |
| `yaw_deg` | float | no | Yaw shorthand when not using pose object |
| `entity_id` | string | no | Semantic entity ID |
| `tags` | string[] | no | Tags for nearest-query |
| `visual_state` | object | no | See §7 |

**Response payload:** `asset_id`, `logical_asset_id`, `position_enu_m`, `position_world_cm`, `rotation_deg`, `actor_name`, `ground_source`.

**Python:** `client.spawn_asset({...})`

---

### 5.2 `simAeroMoveAsset`

Move an existing asset instance.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `asset_id` / `instance_id` | string | **yes** | Instance ID |
| `pose_enu_m` / `pose_world_cm` | object | either | Pose object including position + rotation |
| `position_enu_m` / `position_world_cm` | vector | either | Position shorthand |
| `yaw_deg` | float | no | New yaw |
| `visual_state` | object | no | Updated visual state |

> If the asset's `movement_mode` is `sweep_follow`, UE performs a physics sweep and emits collision feedback on blocking hits.

**Response payload:** `asset_id`, `position_enu_m`, `position_world_cm`, `rotation_deg`, `actor_name`, `ground_source`.

**Python:** `client.move_asset({...})`

---

### 5.3 `simAeroRemoveAsset`

Remove an asset instance.

| Field | Type | Required |
|-------|------|----------|
| `asset_id` | string | **yes** |

**Python:** `client.remove_asset("my_asset_001")`

---

### 5.4 `simAeroReserveOccupancy`

Reserve an asset slot for exclusive use (e.g., landing pad).

| Field | Type | Required |
|-------|------|----------|
| `asset_id` | string | **yes** |
| `entity_id` | string | **yes** |

**Python:** `client.reserve_occupancy("pad_01", "drone_001")`

---

### 5.5 `simAeroReleaseOccupancy`

Release a previously reserved occupancy.

| Field | Type | Required |
|-------|------|----------|
| `asset_id` | string | **yes** |

**Python:** `client.release_occupancy("pad_01")`

---

### 5.6 `simAeroQueryNearest`

Find the nearest asset matching a tag from a given position.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tag` | string | **yes** | Tag to match |
| `pose_enu_m` | vector | no | Query origin |
| `radius_m` | float | no | Search radius |

**Python:** `client.query_nearest("landing_pad", pose_enu_m=(10,20,0))`

**Response payload:** `tag`, `found`, `instance_id`, `logical_asset_id`, `distance_m`, `position_enu_m`, `rotation_deg`, `reserved`, `reserved_by`.

---

## 6. Feedback Event Model

Feedback events are generated by UE when collisions or overlaps occur on managed actors. Each event is tagged with the **frame context** that was active when it was recorded.

### Event Structure

```json
{
  "type": "collision" | "overlap_enter" | "overlap_exit",
  "event_id": "<uuid>",
  "tick": 42,
  "frame_id": 100,
  "episode_id": "ep_001",
  "sample_seq": 5,
  "sim_time_s": 2.1,
  "source_entity_id": "vehicle_001",
  "other_entity_id": "pedestrian_002",
  "source_actor_id": "BP_Vehicle_C_0",
  "other_actor_id": "BP_Ped_C_3",
  "source_logical_asset_id": "vehicle.service.box.v1",
  "other_logical_asset_id": "",
  "source_tags": ["vehicle", "traffic"],
  "other_tags": ["pedestrian"],
  "collision": { ... },
  "overlap": { ... }
}
```

### Collision Detail (`"type": "collision"`)

| Field | Type | Description |
|-------|------|-------------|
| `contact_point_enu_m` | vector | Contact point in ENU metres |
| `contact_normal_enu` | vector | Contact surface normal |
| `relative_speed_mps` | float | Relative speed at impact (m/s) |
| `impulse` | float | Collision impulse |
| `blocking` | bool | Whether the hit was blocking |

### Overlap Detail (`"type": "overlap_enter"` / `"overlap_exit"`)

| Field | Type | Description |
|-------|------|-------------|
| `world_layer_type` | string | Semantic layer of the trigger |
| `zone_kind` | string | Zone kind of the trigger |

### Feedback Modes (per asset template in catalog)

| `feedback_mode` | Behaviour |
|------------------|-----------|
| `none` | No feedback generated |
| `hit` | Collision events only |
| `overlap` | Overlap enter/exit events only |
| `both` | Collision and overlap events |

---

## 7. Visual State

Visual state can be set on spawn or update via the `visual_state` field.

```json
{
  "mode": "visible" | "hidden" | "invisible",
  "variant_id": "variant_a",
  "montage_tag": "idle",
  "lights_on": true,
  "material_variant": "night"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | Visibility mode |
| `variant_id` | string | Visual variant selector |
| `montage_tag` | string | Animation montage tag |
| `lights_on` | bool | Toggle light components |
| `material_variant` | string | Material variant name |

**Current implementation scope:** `mode` (hidden/visible) and `lights_on` are supported on all actors implementing `IAeroVisualStateReceiver`. `variant_id`, `montage_tag`, and `material_variant` are stored but require actor-specific Blueprint implementation to take effect.

---

## 8. Navigation & Utility Operations

### 8.1 `simAeroQueryPedPath`

Query pedestrian navigation path information.

**Python:** `client.query_ped_path(payload)`

---

### 8.2 `simAeroProjectGround`

Project a point onto the ground surface.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `point_enu_m` | vector | **yes** | Point in ENU metres |

**Response payload:** `projected_enu_m`, `surface_normal_enu`, `anchor_id`.

**Python:** `client.project_ground(point_enu_m=(10, 20, 50))`

---

### 8.3 `simAeroQueryPedAnchor`

Query pedestrian anchor point information from the semantic navigation bundle.

**Python:** `client.query_ped_anchor(payload)`

---

### 8.4 `simAeroApplyWeather`

Apply a weather profile or parameters.

**Python:** `client.apply_weather(payload)`

---

## 9. Architecture Summary

```
Python Script (external tick loop)
   │
   ├── AeroSimClient (aero_sim_client.py)
   │      │
   │      ├── simAeroApplyFrame ───► UAeroBridgeWorldSubsystem
   │      │                              │
   │      │                              ├── UAeroSceneSyncSubsystem::ApplyFrame
   │      │                              │     └── UAeroAssetPlacementSubsystem::SpawnOrUpdateProxy
   │      │                              │
   │      │                              └── UAeroFeedbackSubsystem::SetFrameContext
   │      │
   │      ├── simAeroPollFeedback ──► UAeroFeedbackSubsystem::PollFeedbackSinceTick
   │      │
   │      ├── simAeroPed* ──────────► UAeroRuntimeOrchestrationSubsystem
   │      │
   │      └── simAeroSpawn/Move/RemoveAsset ──► UAeroAssetPlacementSubsystem
   │
   └── Reads asset_catalog.json (to know available proxy_template_ids)

UE Engine (render + physics)
   │
   ├── UAeroCollisionRelayComponent ──► UAeroFeedbackSubsystem::EnqueueFeedback
   ├── UAeroTriggerRelayComponent ───► UAeroFeedbackSubsystem::EnqueueFeedback
   └── SweepFollow movement mode ───► MaybeEmitSweepCollision
```

---

## 10. `asset_catalog.json` Template Fields

Each entry in `asset_catalog.json` defines a spawnable template:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `logical_asset_id` | string | **yes** | Unique template identifier |
| `semantic_type` | string | **yes** | Category: `vehicle`, `pedestrian`, `prop`, `trigger`, `facility`, `uav`, `infrastructure` |
| `spawn_backend` | string | **yes** | `ue_actor`, `trigger_zone`, `semantic_only` |
| `ue_asset_path` | string | no | UE Blueprint or mesh path |
| `is_blueprint` | bool | no | Whether `ue_asset_path` is a Blueprint class |
| `airsim_registry_name` | string | no | AirSim vehicle registry name |
| `collision_profile` | string | no | UE collision profile name |
| `feedback_mode` | string | no | `none`, `hit`, `overlap`, `both` |
| `movement_mode` | string | no | `teleport` (default), `sweep_follow` |
| `ground_snap_policy` | string | no | `project_down`, or empty |
| `default_scale_xyz` | vector | no | Scale override |
| `default_yaw_offset_deg` | float | no | Yaw offset applied on spawn |
| `default_z_offset_m` | float | no | Vertical offset in metres |
| `physics_enabled` | bool | no | Enable UE physics simulation |
| `world_layer_type` | string | no | Semantic layer for triggers |
| `zone_kind` | string | no | Semantic zone kind for triggers |
| `label_class` | string | no | Annotation/label class |
| `render_required` | bool | no | Whether a visible actor is needed |
| `annotation_visible` | bool | no | Visible in annotation pass |
| `reservable` | bool | no | Can be reserved via `simAeroReserveOccupancy` |
| `blocking` | bool | no | Whether the asset blocks movement |
| `query_tags` | string[] | no | Default tags for nearest-query |
| `default_visual_state` | object | no | Default visual state (see §7) |

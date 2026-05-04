# AeroWorldContent

This plugin is the home for UE assets and authored content used by scenarios.

If something is a UE asset rather than Python-side data, it should usually live here.

## Read Order

1. `README.md`
   Ownership and asset placement rules.
2. `SCENARIO_ASSET_LAYOUT.md`
   Incident-material layout for reusable and scenario-specific props.

## What Belongs Here

- Blueprints for triggers, facilities, vehicles, and UAV support actors
- Data assets for pedestrian and crowd configuration
- Meshes and materials for props, facilities, infrastructure, and scenario visuals
- Reusable incident assets such as flower pots, cones, umbrellas, warning props, and street-scene assets

## Suggested Locations

- `Content/Blueprints/Triggers/`
- `Content/Blueprints/Facilities/`
- `Content/Blueprints/Vehicles/`
- `Content/Blueprints/UAV/`
- `Content/DataAssets/Ped/`
- `Content/DataAssets/Crowd/`
- `Content/Props/`

## Recommended Asset Names

- `BP_AW_Trigger_NoFly_Box_01`
- `BP_AW_Trigger_Hazard_Construction_Box_01`
- `BP_AW_Trigger_Hazard_Generic_Box_01`
- `DA_AW_PedVariants_CityOps_01`
- `DA_AW_CrowdAppearancePool_CityOps_01`
- `DA_AW_CrowdRoleProfile_CityOps_Default_01`

## Boundary

Do not put these here:

- Python scenario specs
- truth JSONL outputs
- capture plans
- offline map-build source files

Those belong in `Plugins/SumoImporter`.

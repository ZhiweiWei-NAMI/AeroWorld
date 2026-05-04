# Donghu Map Sources

This directory owns the plugin-managed source inputs for `donghu_road_topo`.

## What Lives Here

- `source/map.net.xml`
- `source/map.osm`
- `source/temp.net.xml`

These are upstream map-build inputs and temporary conversion artifacts used by Python-side workflows.

## What Does Not Live Here

- UE runtime map config does not currently live here.
  `AeroSimHost` resolves runtime map files from `Config/LowAltitude/Maps/donghu_road_topo/`, so files such as:
  `map_context.json`
  `ped_nav_semantic.bundle.json`
  `ped_nav_semantic.source.json`
  `scenario_objects.json`
  `traffic_bundle/*`
  still stay in `Config` for runtime compatibility.

## Rule

If a map file is a source/build artifact and not directly loaded by UE runtime code, keep it under this plugin-owned directory instead of the repo root.

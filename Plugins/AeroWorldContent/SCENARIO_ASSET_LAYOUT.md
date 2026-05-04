# Scenario Asset Layout

`AeroWorldContent` owns UE assets used by scenarios.

## Put Reusable Incident Props Here

- `Content/Props/ScenarioCommon/FallingObjects/`
  Flower pots, tools, boxes, loose rooftop objects, and other generic falling-object assets.

- `Content/Props/ScenarioCommon/StreetIncidents/`
  Barricades, warning signs, cones, tape, umbrellas, delivery bags, and similar reusable street-scene props.

## Put Scene-Specific Assets Here

- `Content/Props/ScenarioSpecific/<scenario_id>/`
  Assets that are only meaningful for one scenario family and are not expected to be reused broadly.

## What Should Not Go Here

- Python scenario specs
- capture plans
- truth JSONL outputs
- offline map-build sources

Those belong in `Plugins/SumoImporter`.

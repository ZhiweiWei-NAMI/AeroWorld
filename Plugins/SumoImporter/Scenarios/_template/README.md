# Scenario Template

Copy this structure when creating a new scenario family.

## Expected Subdirectories

- `spec/`
  Source-of-truth scenario definition.

- `scripts/`
  Scenario-specific build and validation scripts.

- `artifacts/`
  Generated outputs for the built scenario package.

- `notes/`
  Manual notes, review comments, and operating constraints.

## Example Materials Split

- Python-side scenario metadata:
  keep in this scenario folder

- UE assets such as a flower pot used by a falling-object scenario:
  place in `Plugins/AeroWorldContent/Content/Props/ScenarioCommon/FallingObjects/`
  or `ScenarioSpecific/<scenario_id>/` if the asset is unique

# UAM Dataset Coverage Report

## Summary

- **Total scenario definitions**: 70
- **Base scenarios**: 64 across L1-L6
- **Cross-layer chains**: 6 under `X_cross_layer`
- **Event steps**: 257
- **Event types covered**: 33/33 (100%)
- **CAAC emergency events**: 12/14 (85.7%, 2 out of scope)
- **ODD layers**: 6/6 (100%)
- **Mechanism categories**: 5/5 (100%)
- **Scene grounding validation**: 70/70 pass with `Dataset/tools/validate_scene_grounding.py`

## Per-Layer Breakdown

| Layer | Event Types | Scenario definitions | Main entities |
|-------|-------------|----------------------|---------------|
| L1 Airspace | 4 (L1-1 to L1-4) | 7 | UAV, no-fly zone, corridor/hazard zone |
| L2 Infrastructure | 5 (L2-1 to L2-5) | 9 | UAV, radio tower, charger, landing pad, traffic signal, vehicle |
| L3 Dynamic Constraints | 3 (L3-1 to L3-3) | 5 | roadwork props, hazard zone, UAV, vehicle, pedestrian |
| L4 Agents | 11 (L4-1 to L4-11) | 24 | UAV, vehicle, pedestrian, crowd |
| L5 Environment | 5 (L5-1 to L5-5) | 9 | UAV, vehicle, pedestrian, weather |
| L6 Digital Layer | 5 (L6-1 to L6-5) | 10 | UAV, radio tower, ground station |
| X Cross-Layer | 6 chains | 6 | mixed entities spanning at least two ODD layers |

## Cross-Layer Event Chains

| Chain | Description | Layers | Event steps |
|-------|-------------|--------|-------------|
| X1 | Rain -> C2 loss -> forced landing | L5 -> L6 -> L4 | 6 |
| X2 | GNSS spoofing -> geofence violation | L6 -> L1 | 5 |
| X3 | Pedestrian fall -> emergency response chain | L4 -> emergency response | 5 |
| X4 | Fog -> UAV-UAV conflict | L5 -> L4 | 5 |
| X5 | Communication failure -> pad contention | L2 -> L2 | 5 |
| X6 | Crowd evacuation -> airspace lockdown | L4 -> L3 | 5 |

## Validation Scope

`validate_coverage.py` checks taxonomy coverage. `validate_scene_grounding.py` checks executable grounding:

- event-script entity references resolve to `scene_setup.entities`;
- logical asset IDs exist in `Config/LowAltitude/asset_catalog.json`;
- lane, sidewalk, crosswalk, facade, pad, box, and polygon placements expose concrete ENU positions;
- roadwork props are outside lane center and on one shoulder side;
- pedestrian and crowd positions stay out of roadway except explicit crossing/retreat phases;
- weather-state scenarios have tick/weather-profile bootstraps;
- composite triggers reference existing child triggers;
- scenario `validation_rules` are executed and reported pass/fail.

## CAAC Compliance

| CAAC | Status | Notes |
|------|--------|-------|
| CAAC-1 (manned aircraft collision) | Out of scope | No manned aircraft in low-altitude UAM ODD |
| CAAC-2 (crash/forced landing) | Covered | L4-3, X1 |
| CAAC-3 (loss of control) | Covered | L4-1, L5-3, L6-3 |
| CAAC-4 (obstacle strike) | Covered | L4-2, L5-1, L5-2 |
| CAAC-5 (runway excursion) | Out of scope | Requires vertiport runway infrastructure |
| CAAC-6 (takeoff/landing errors) | Covered | L2-4 |
| CAAC-7 (fire/smoke) | Covered | L4-3 related |
| CAAC-8 (propulsion/electrical failure) | Covered | L4-3, L6-1 |
| CAAC-9 (C2 link loss) | Covered | L6-1, L6-2, X1 |
| CAAC-10 (prohibited zone entry) | Covered | L1-1, L1-3, L3-2, X2 |
| CAAC-11 (emergency evacuation) | Covered | L4-8, X6 |
| CAAC-12 (emergency response at sites) | Covered | L2-4, L3-3, X3 |
| CAAC-13 (collision with facilities/vehicles/personnel) | Covered | L4-2, L4-4, L4-5 |
| CAAC-14 (death/serious injury) | Covered | L4-3, L4-5, L4-6 |

## Standards Alignment

- **NASA Belcastro et al. (2017)**: all 4 hazard domains covered.
- **JARUS SORA v2.5**: SAIL I through VI represented.
- **ICAO CICTT (ADREP)**: MAC, GCOL, CFIT, SCF-NP, SCF-PP, NAV, SEC, WSTRW, TURB, UIMC, MED, and EVAC represented.
- **PEGASUS 6-layer model**: all UAM-adapted layers represented with multiple scenario definitions.

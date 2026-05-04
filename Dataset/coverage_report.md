# UAM Dataset Coverage Report

## Summary

- **Total scenarios**: 70 (64 P0/P1 + 6 cross-layer chains)
- **Event types covered**: 33/33 (100%)
- **CAAC emergency events**: 12/14 (85.7%, 2 out of scope)
- **ODD Layers**: 6/6 (100%)
- **Mechanism categories**: 5/5 (100%)

## Per-Layer Breakdown

| Layer | Event Types | Scripts | Entities |
|-------|-----------|---------|----------|
| L1 Airspace | 4 (L1-1 to L1-4) | 7 | UAV, no_fly_zone, corridor_zone |
| L2 Infrastructure | 5 (L2-1 to L2-5) | 9 | UAV, radio_tower, charger, landing_pad, traffic_light, vehicle |
| L3 Dynamic Constraints | 3 (L3-1 to L3-3) | 6 | barrier, hazard_zone, UAV, vehicle, pedestrian |
| L4 Agents | 11 (L4-1 to L4-11) | 24 | UAV×2-3, vehicle×2, pedestrian, crowd |
| L5 Environment | 5 (L5-1 to L5-5) | 9 | UAV, vehicle, pedestrian |
| L6 Digital Layer | 5 (L6-1 to L6-5) | 10 | UAV, radio_tower, operator |
| X Cross-Layer | 6 chains | 6 | Mixed (spanning 2-3 layers each) |

## Cross-Layer Event Chains

| Chain | Description | Layers | Events |
|-------|------------|--------|--------|
| X1 | Rain → C2 loss → Forced landing | L5→L6→L4 | 3 |
| X2 | GPS spoofing → Geofence violation | L6→L1 | 2 |
| X3 | Pedestrian fall → Ambulance → Isolation zone | L4→L3 | 4 |
| X4 | Fog → UAV-UAV conflict | L5→L4 | 3 |
| X5 | Station failure → C2 loss → Pad contention | L2→L6→L2 | 3 |
| X6 | Crowd evacuation → NFZ activation | L4→L3 | 2 |

## CAAC Compliance

| CAAC | Status | Notes |
|------|--------|-------|
| CAAC-1 (manned aircraft collision) | Out of scope | No manned aircraft in low-altitude UAM ODD |
| CAAC-2 (crash/forced landing) | Covered | L4-1, L4-3 |
| CAAC-3 (loss of control) | Covered | L4-1, L5-3, L6-3 |
| CAAC-4 (obstacle strike) | Covered | L4-2, L5-1, L5-2 |
| CAAC-5 (runway excursion) | Out of scope | Requires vertiport runway infrastructure |
| CAAC-6 (takeoff/landing errors) | Covered | L2-4 |
| CAAC-7 (fire/smoke) | Covered | L4-3 related |
| CAAC-8 (propulsion/electrical failure) | Covered | L4-3, L6-1 |
| CAAC-9 (C2 link loss) | Covered | L6-1, L6-2 |
| CAAC-10 (prohibited zone entry) | Covered | L1-1, L1-3, L3-2 |
| CAAC-11 (emergency evacuation) | Covered | L4-8 |
| CAAC-12 (emergency response at sites) | Covered | L2-4, L3-3 |
| CAAC-13 (collision with facilities/vehicles/personnel) | Covered | L4-2, L4-4, L4-5 |
| CAAC-14 (death/serious injury) | Covered | L4-3, L4-5, L4-6 |

## Standards Alignment

- **NASA Belcastro et al. (2017)**: All 4 hazard domains covered; 27 of 31 sub-categories mapped
- **JARUS SORA v2.5**: SAIL I through VI all represented
- **ICAO CICTT (ADREP)**: MAC, GCOL, CFIT, SCF-NP, SCF-PP, NAV, SEC, WSTRW, TURB, UIMC, MED, EVAC — all represented
- **PEGASUS 6-layer** (UAM-adapted): All 6 layers with at least 4 scripts each

# PedestrianRuntime

This plugin owns UE-side pedestrian behavior after the bridge hands off a managed pedestrian.

## Read Order

1. `README.md`
2. `Source/PedestrianRuntime/Public/PedestrianWorldSubsystem.h`
3. `Source/PedestrianRuntime/Private/PedestrianWorldSubsystem.cpp`
4. `Source/PedestrianRuntime/Private/GroundPlacementUtils.cpp`
5. `Source/PedestrianRuntime/Private/PedestrianVariantCatalog.cpp`

## Responsibilities

- managed pedestrian spawning and reset behavior
- grounding and placement helpers
- variant catalog and appearance selection
- pedestrian runtime subsystem behavior
- crowd spawning helpers

## Design Idea

Current project direction is:

- truth remains authoritative for pedestrian pose in Python-driven scenarios
- UE-side managed pedestrians still provide animation, representation, and grounding support
- incident states such as falls should preserve the intended incident pose and not drift away from the event point

## When To Change This Plugin

- change grounding behavior for pedestrians
- change animation or montage behavior for managed pedestrians
- change variant selection or crowd appearance behavior
- change UE-side runtime logic after `simAeroPed*` requests are accepted

## Important Interaction

Python code in `SumoImporter` calls bridge APIs.
`AeroSimHost` forwards pedestrian actions into runtime subsystems.
`PedestrianRuntime` is where the UE-side pedestrian behavior actually lives.

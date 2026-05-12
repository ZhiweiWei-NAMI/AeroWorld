# Handover: Low-Altitude Semantic Event Chain

Date: 2026-05-12
Workspace: `E:\DynamicCityCreatorSamples`
Supersedes: `HANDOVER_SKY_DOME_SEMANTIC_CAPTURE.md`

## Current State

The project has already moved past contract design and into concrete regeneration, but it is not yet safe to begin the canonical dozens-of-hours capture run.

| Area | Status | Evidence | What blocks the next step |
|---|---|---|---|
| Contract centralization | Done | `Dataset/tools/semantic_event_contract.py` now holds the canonical low-altitude semantic event-chain contract | Validator semantics still need alignment |
| Scenario regeneration | Done | All 70 scenario directories were regenerated with updated `spec.py`, `scene_setup.json`, and `event_script.json` | Some regenerated episodes still fail validator checks |
| Background semantic actors | Done | Background vehicles and pedestrians are now injected deterministically as semantic actors | Need truth-frame and render-ready verification |
| `U_inspect` long-lived motion | Done | `U_inspect` now has a long route and is no longer treated as a static hover actor | Corridor validation still misclassifies it in some cases |
| Reachability validation | Blocked | `python Dataset/tools/validate_event_reachability.py` reports 61 issues across 70 scenarios | Semantic alias / intent mapping is incomplete |
| Grounding validation | Blocked | `python Dataset/tools/validate_scene_grounding.py` reports 1463 issues | `U_inspect` and observer UAVs are still checked like ordinary high-altitude mission UAVs |
| Long-run capture | Not started | `UnrealEditor.exe` is still running, but no canonical long-run capture job is active yet | Validators must be green first |
| Coverage validation | Pass | `validate_coverage.py` passes | None |

Checkpoint commit at the moment of this handoff: `77c02c2` (`Checkpoint semantic event-chain contract`).

## Actual Progress

- The canonical pipeline has been reduced to one path only:
  `spec_compiler.py -> regenerate_boundary_scenarios.py -> batch_generate.py -> convert_to_render_ready.py -> run_semantic_event_chain_every10.py -> episode_render_host.py -> validators`
- The old batch / AutoPIE style paths are not part of the plan.
- The session has already established that the project goal is not just “render something”, but to make every scene entity physically active and semantically linked.
- `U_inspect` is now treated as a long-lived inspect-view substitute, not a static hover token.
- Background `V/P` are now part of the semantic contract, not decorative filler.
- The current blocker is not the absence of data; it is validator contract mismatch and a smaller set of true trajectory issues.

## What Is Blocked

| Blocker | Why it blocks | Concrete symptom | Prejudged resolution |
|---|---|---|---|
| Literal event matching | It is too strict for semantic phrasing | `validate_event_reachability.py` misses `RTH`, `avoid`, `deconflict`, `failsafe`, `slowdown`, `recovery` variants | Add explicit alias / intent mapping before order checking |
| Single corridor rule for all UAVs | It treats `U_inspect` and observer UAVs as ordinary mission UAVs | `validate_scene_grounding.py` produces many `UAV mission/start point is outside high-altitude corridors` errors | Split the corridor contract by role, while keeping the shared safety base rules |
| Real trajectory defects | Not every failure is a validator artifact | Some scenarios still have geometry issues such as diversion starts that are too far from the current state | Regenerate the affected trajectories after the validator contract is fixed |

The key interpretation is this: the current `U_inspect` corridor handling is **not yet compatible** with the existing high-altitude mission validator as written. That is a design mismatch, not a reason to add a fallback. The fix is to define the inspect-view corridor contract explicitly, then keep the shared safety invariants that both inspect and mission UAVs must satisfy.

## Core Goal

Make every entity in every episode physically move and semantically participate in the episode:

- every UAV must move, including `U_inspect` and any observer UAVs
- every background vehicle and pedestrian must have a semantic role and a state
- every episode must contain continuous interaction, not isolated props
- every episode must contain one unmistakable key event
- the truth-frame side and the UE visual side remain decoupled, but they must agree on the same semantic chain

The design principle for this round is: take the highest target that is still deterministic. Do not downgrade to a weaker path when the intended contract is still achievable. If a higher target cannot be satisfied deterministically, stop and report the exact missing invariant.

## Execution Boundary

- Deterministic only.
- No fallback.
- No guessing.
- No compatibility path.
- No legacy runner.
- No silent downgrade to a weaker plan.
- No closing UE/PIE except for C++ rebuilds or explicit user request.
- No new timestamped / versioned output roots for normal runs.
- No `plan_capture_schedule.py`, `run_auto_pie_capture.py`, or `batch_render_dataset.py` path in the canonical workflow.
- Background vehicles and pedestrians are semantic actors, not decoration.
- `event_trace.jsonl`, `dynamic_labels.jsonl`, and `truth_frames.jsonl` must remain source-consistent.
- If an invariant cannot be satisfied, stop and report the exact missing invariant rather than inventing a substitute.

## Canonical Plan

1. Normalize event semantics.
   - Add explicit alias / intent mapping in `semantic_event_contract.py` for required event phrases.
   - Preserve the existing stage ordering logic, but make the semantic matching robust enough to accept intentional phrasing variants.

2. Split UAV corridor logic by role.
   - Give `U_inspect` its own inspect corridor / altitude rule.
   - Keep the shared safety constraints that all UAVs still must satisfy.
   - Make the design decision explicit if the current high-altitude rule family cannot be reused as-is.

3. Fix true trajectory defects.
   - Re-run the scenario generator only after the validator contract is updated.
   - Repair the episodes with genuine path defects instead of papering over them.

4. Rebuild render-ready outputs.
   - Ensure `scene_setup.json`, `event_script.json`, `truth_frames.jsonl`, `event_trace.jsonl`, and `dynamic_labels.jsonl` agree.
   - Ensure capture-ready artifacts remain deterministic and aligned to the canonical chain.

5. Start canonical capture only after validators are clean.
   - Keep UE open.
   - Reuse the existing PIE session and AirSim RPC.
   - Run the long capture job for dozens of hours once the invariant set is clean.

## Subagent Workstream

The documentation work is being split into three parallel scopes:

- root `Dataset` docs
- per-episode `Dataset/scenarios/**/README.md` docs
- `Plugins/SumoImporter/**.md` runtime and layout docs

All subagent prompts must preserve the same rule:

`100% deterministic only. No fallback. No guessing. No compatibility path. Background vehicles and pedestrians are semantic actors, not decoration. Preserve every known deterministic event, count, task, state, and file boundary. If an invariant cannot be satisfied, stop and report the exact missing invariant.`

## Session Notes

- The current root handoff replaces the older sky-dome / semantic-capture wording with the actual low-altitude semantic event-chain contract.
- The project now has enough material to explain the scene contract clearly, but not enough validator convergence to start uninterrupted long capture.
- The truth-frame and render layers are intentionally decoupled. That is correct design, but it makes visual audit mandatory.
- A few remaining trajectory issues are genuine and will need regeneration after semantic validation is fixed.

## Episode Acceptance Matrix

This matrix is the locked episode contract for later implementation and review. The exact counts and semantic roles below are the acceptance target, not a suggestion.

| Episode | Exact U/V/P/F/L | Inspect | Required Event | Background V/P Semantic Role | Weather |
|---|---:|---|---|---|---|
| L1-1_v1 | 3/2/2/2/9 | I28 | boundary conflict > avoid/RTH > land | V: normal road flow under airspace boundary; P: walkers showing inhabited corridor | clear |
| L1-1_v2 | 3/2/2/2/7 | I28 | alternate geofence intrusion > RTH > land | V: steady traffic; P: waiting/walking near visible ground reference | clear |
| L1-2_v1 | 3/2/2/2/8 | I28 | altitude deviation > correction > land | V: moving ground scale; P: sidewalk walkers below altitude corridor | clear |
| L1-3_v1 | 3/2/2/3/10 | I28 | intruder conflict > deconflict > dual landing | V: uninterrupted flow under conflict; P: bystanders not reacting unless risk descends | clear |
| L1-3_v2 | 3/2/2/3/10 | I28 | fast intruder crossing > separation > land | V: moving road context; P: pedestrian flow giving low-altitude reference | clear |
| L1-4_v1 | 4/2/2/4/15 | I28 | congestion > priority resequence > land | V: slow background traffic; P: waiting near facility/pad area | clear |
| L1-4_v2 | 4/2/2/4/15 | I28 | dense congestion > hold/divert > staged landings | V: queued/slow traffic; P: facility-area walking/waiting | clear |
| L2-1_v1 | 3/3/2/3/8 | I18 | tower degraded > backup restore > land | V: traffic continues during C2 degradation; P: facility pedestrians near tower | clear |
| L2-1_v2 | 3/3/2/3/5 | I18 | C2 degraded > hold/reroute > restore | V: moving traffic; P: pedestrians provide facility context | clear |
| L2-2_v1 | 3/3/2/3/6 | I18 | GNSS anomaly > drift > relocalize | V: road flow used as visual relocalization background; P: sidewalk reference | clear |
| L2-2_v2 | 3/3/2/3/6 | I18 | longer GNSS drift > correction | V: moving lane references; P: non-reactive walkers | clear |
| L2-3_v1 | 3/2/2/3/6 | I18 | charger unavailable > backup landing | V: facility access traffic; P: pedestrians near charging area | clear |
| L2-3_v2 | 3/2/2/3/6 | I18 | alternate charger failure > reroute | V: access-road background; P: waiting/walking near charger | clear |
| L2-4_v1 | 3/2/2/3/8 | I18 | pad request > arbitration > divert | V: service vehicles near pad; P: facility waiting context | clear |
| L2-4_v2 | 3/2/2/3/8 | I18 | pad contention > priority landing | V: service-road flow; P: ground observers near pad | clear |
| L2-5_v1 | 3/5/4/2/3 | I18 | signal fault > queue > manual flow | V: queue/yield/manual-control traffic; P: crosswalk wait/cross | clear |
| L3-1_v1 | 3/3/3/2/14 | I18 | roadwork closure > detour > inspect | V: detour/blocked/slow vehicles; P: pedestrians reroute around barriers | clear |
| L3-2_v1 | 3/2/2/2/6 | I28 | NFZ proximity alert > reroute | V: ordinary traffic beneath NFZ; P: ground population context | clear |
| L3-2_v2 | 3/2/2/2/6 | I28 | alternate NFZ proximity > reroute | V: moving traffic; P: walkers giving occupied-zone context | clear |
| L3-3_v1 | 3/2/8/4/8 | I18 | hazmat leak > isolation > evac | V: ambulance/service response; P: evacuation cohort | clear |
| L3-3_v2 | 3/2/8/4/8 | I18 | hazmat spread > evacuation > handoff | V: responder + blocked traffic; P: evacuating and waiting groups | clear |
| L4-1_v1 | 3/2/2/2/12 | I28 | UAV convergence > separation | V: traffic below conflict area; P: passive urban context | clear |
| L4-1_v2 | 3/2/2/2/12 | I28 | alternate convergence > deconflict | V: steady flow; P: sidewalk walkers | clear |
| L4-2_v1 | 3/2/2/2/6 | I18 | facade proximity > evade | V: road context near facade; P: building-side pedestrians | clear |
| L4-2_v2 | 3/2/2/2/6 | I18 | facade near miss > recovery | V: moving/stopped near building; P: facade-scale pedestrians | clear |
| L4-3_v1 | 3/2/8/2/4 | I10 | forced descent > crowd response | V: hold/blocked by landing zone; P: evade/retreat crowd | clear |
| L4-3_v2 | 3/2/8/2/4 | I10 | forced landing > crowd clear | V: emergency hold; P: clear landing path | clear |
| L4-3_v3 | 3/2/8/2/4 | I10 | forced landing variant > touchdown | V: stopped/slow response; P: bystanders retreat | clear |
| L4-4_v1 | 3/3/2/1/4 | I18 | UAV-vehicle crossing > brake | V: brake/yield/contact-risk traffic; P: roadside witnesses | clear |
| L4-4_v2 | 3/3/2/1/4 | I18 | crossing > emergency stop | V: emergency stop/following vehicle reaction; P: waiting context | clear |
| L4-5_v1 | 3/2/4/1/5 | I10 | pedestrian near-miss > pull-up | V: nearby slow traffic; P: target + retreating pedestrians | clear |
| L4-5_v2 | 3/2/4/1/5 | I10 | alternate near-miss > clear | V: road context; P: clear/retreat/wait states | clear |
| L4-5_v3 | 3/2/4/1/5 | I10 | low-alt inspect > near-miss recovery | V: moving context; P: inspected group reacts | clear |
| L4-6_v1 | 3/2/3/1/2 | I10 | jaywalk > vehicle brake > retreat | V: braking/yielding vehicles; P: jaywalker + waiting peds | clear |
| L4-6_v2 | 3/2/3/1/2 | I10 | alternate jaywalk conflict | V: brake/recover traffic; P: retreat/wait | clear |
| L4-7_v1 | 3/2/4/1/3 | I10 | fall > UAV detect > ambulance | V: ambulance/yield traffic; P: fallen + bystanders | clear |
| L4-7_v2 | 3/2/4/1/3 | I10 | fall response > medical handoff | V: responder vehicle and yielding car; P: bystanders/medical wait | clear |
| L4-8_v1 | 3/2/12/2/4 | I10 | crowd evacuation > safe hold | V: stopped/held perimeter vehicles; P: evacuating crowd | clear |
| L4-8_v2 | 3/2/12/2/8 | I10 | evacuation variant > land | V: perimeter hold; P: evacuation to safe zone | clear |
| L4-9_v1 | 3/3/3/1/2 | I18 | vehicle conflict > warning/brake | V: conflict/brake/yield chain; P: waiting at roadside | clear |
| L4-9_v2 | 3/3/3/1/2 | I18 | alternate vehicle conflict | V: lane conflict/recovery; P: scale and risk background | clear |
| L4-10_v1 | 3/4/3/1/3 | I18 | ambulance priority > yield | V: ambulance priority + civilian yield; P: crosswalk wait | clear |
| L4-10_v2 | 3/4/3/1/3 | I18 | ambulance priority > clearance | V: queue/yield/clearance; P: waiting pedestrians | clear |
| L4-11_v1 | 3/3/2/1/3 | I18 | AV fault > safe stop > report | V: AV stop + follower reaction; P: roadside context | clear |
| L4-11_v2 | 3/3/2/1/3 | I18 | AV failure > hazard hold | V: stopped AV and following traffic; P: non-event background | clear |
| L5-1_v1 | 3/4/6/2/6 | I22 | rain > slowdown > recovery | V: rain-slow traffic; P: seek shelter/walk slower | rain |
| L5-1_v2 | 3/4/6/2/6 | I22 | rain variant > degraded route | V: cautious traffic; P: shelter/wait states | rain |
| L5-1_v3 | 3/4/6/2/6 | I22 | heavy rain > safe land | V: slow/queued; P: sheltering and crossing delay | rain |
| L5-2_v1 | 3/3/6/2/6 | I22 | fog onset > abort > land | V: cautious low-visibility traffic; P: slow/wait | fog |
| L5-2_v2 | 3/3/6/2/6 | I22 | fog variant > recovery | V: slow traffic; P: reduced-visibility walking | fog |
| L5-3_v1 | 3/3/4/2/6 | I22 | wind > payload swing > recovery | V: normal flow; P: wind-affected walking/waiting | wind |
| L5-3_v2 | 3/3/4/2/6 | I22 | gust variant > land | V: moving context; P: cautious movement | wind |
| L5-4_v1 | 3/3/4/2/5 | I22 | dusk > IR switch > validate | V: low-light traffic; P: low-light crossing/wait | dusk |
| L5-5_v1 | 3/3/4/2/5 | I22 | heat derate > charger decision | V: normal/slowed access traffic; P: heat-wait/shelter | heat |
| L6-1_v1 | 3/3/4/3/8 | I22 | C2 loss > failsafe/RTH | V: unaffected traffic proving digital-only fault; P: operator-area context | clear |
| L6-1_v2 | 3/3/4/3/8 | I22 | long-route C2 loss > recovery | V: steady traffic; P: facility pedestrians | clear |
| L6-2_v1 | 3/3/4/3/8 | I22 | latency > backup link | V: traffic continues; P: passive background near facility | clear |
| L6-2_v2 | 3/3/4/3/8 | I22 | packet loss > recovery | V: moving road context; P: sidewalk context | clear |
| L6-3_v1 | 3/3/4/3/8 | I22 | spoof > offset > correction | V: street reference for route offset; P: occupied geofence context | clear |
| L6-3_v2 | 3/3/4/3/8 | I22 | spoof variant > recovery | V: moving landmarks; P: ground context | clear |
| L6-4_v1 | 3/3/4/3/10 | I22 | jamming > safe hold > channel | V: unaffected traffic; P: facility/road background | clear |
| L6-4_v2 | 3/3/4/3/10 | I22 | jamming variant > land | V: road context; P: passive scale/occupancy | clear |
| L6-5_v1 | 3/3/4/3/8 | I22 | GCS intrusion > lockout | V: traffic below abnormal path; P: operator-area context | clear |
| L6-5_v2 | 3/3/4/3/8 | I22 | intrusion variant > secure recovery | V: steady context; P: facility pedestrians | clear |
| X1 | 3/3/10/3/8 | I10 | rain + C2 loss > forced landing | V: emergency response/hold; P: crowd evade/recover | rain |
| X2 | 3/3/4/2/8 | I22 | spoof > NFZ violation alert | V: street reference and risk context; P: occupied zone context | clear |
| X3 | 3/3/6/2/6 | I10 | fall > responder > handoff | V: ambulance/yield chain; P: fallen + bystanders | clear |
| X4 | 3/3/4/2/10 | I22 | fog > UAV conflict > evasion | V: low-visibility traffic; P: fog-context pedestrians | fog |
| X5 | 3/3/4/5/10 | I18 | station fail > pad contention | V: facility access traffic; P: waiting near facility | clear |
| X6 | 3/3/10/3/12 | I10 | crowd evac > NFZ lockdown | V: perimeter vehicles move/hold; P: evacuation to safe zone | clear |

## Immediate Next Actions

1. Finish the semantic alias / intent layer so reachability validation can distinguish phrasing from meaning.
2. Split `U_inspect` into its own corridor contract while preserving the shared safety base rules.
3. Re-run the generators only after the validators have a coherent contract.
4. Start the canonical capture runner only when the event chain, grounding, and truth-frame checks all converge.

## Notes For Future Agents

- The session intent is to avoid any second path or fallback path.
- The goal is not merely scene variety; it is a physically animated, semantically coherent, multi-entity low-altitude event chain.
- The documentation work is currently being delegated to parallel docs_writer agents for the root Dataset docs, per-scenario README files, and SumoImporter docs.
- Keep the same strict boundary when editing any future code or docs: if an invariant cannot be proven, do not hide it.

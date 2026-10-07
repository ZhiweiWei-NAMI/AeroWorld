"""First abnormal-action execution hook for the L6-5_v1 single-action prefix.

Approved scope (native v6 checkpoint, this module only): integrate the ONE
authored first abnormal movement action
(``move_uav_intrusion_abnormal`` of event ``abnormal_uav_behavior``) with the
existing frozen receipt/authority gates, using the EXISTING
``Dataset.semantic_simulation.ns3_episode.linked_engine.execute_linked``
callbacks. No new transport, no new framework, no scheduler change: the
engine's current tick clock is authoritative and is used as the executor's
current ``execution_ns`` (``tick * adapter.STEP_NS``), never a forced grid.

How the hook attaches (exact existing semantics, verified from source):

- ``decision_gate(event_def, tick, engine, interpreter)`` — the engine's
  ``AdmittedInterpreter`` consults it per ``(event_id, tick)`` after the base
  max-fire guard and requires exactly ``{'allow': bool, 'evidence': ...}``.
  The engine cache is per tick, so a deferral retries the NEXT tick and never
  consumes the event's ``max_fire_count``. This prefix defers the target
  event until the correct native receipt is actually available at the current
  tick (admitted through the current adapter identity/control/time contract)
  and passes every other event through unchanged.
- ``action_filter(action, clock_tick)`` — the engine calls it on a deepcopy of
  the resolved action immediately before the target move handler. A ``None``
  return is audited by the engine as ``omitted_by_explicit_configuration``
  and the handler is never reached. Immediately before the handler, this
  prefix runs the REAL pure readiness check
  (``adapter.check_movement_execution_ready``) at the dispatcher's current
  instant and passes the ORIGINAL resolved move through unchanged ONLY when
  it is ready.

Evidence separation: gate evidence may carry the pure readiness record, which
always has ``applied=False`` (it is a CHECK, never an execution). Actual
dispatch evidence comes only from ``execute_linked``'s own audit/result of the
real handler, never from this gate.

Prefix bounds: the production profile0 epoch0 grant is initially active for
this run/UAV/incident, and this checkpoint wires nothing else — no lockout,
no recovery/notification, no landing, no future story content. Identical
current inputs therefore produce identical prefix decisions whether or not a
future story exists. Posthoc validation (movement-authority state, lockout
report, motion binding) is separate and unchanged.
"""
from __future__ import annotations

import copy

from Dataset.semantic_simulation.p09_l6_5_receipt_v1 import adapter as A

TARGET_EVENT_ID = "abnormal_uav_behavior"
TARGET_ACTION_ID = A.MOVEMENT_ACTION  # "move_uav_intrusion_abnormal"


def default_readiness_check(binding, receipt_record, *, execution_ns,
                            movement_epoch0_active):
    """The real adapter readiness check (explicit default, no hidden logic)."""
    return A.check_movement_execution_ready(
        binding, receipt_record, execution_ns=execution_ns,
        movement_epoch0_active=movement_epoch0_active)


def prefix_action(action, *, tick, binding, receipt_record,
                  movement_epoch0_active=True, readiness_check=None,
                  refusals=None):
    """Dispatcher-side final gate immediately before the target move handler.

    Identity first, then readiness. The engine routes EVERY dispatched action
    through its single ``action_filter``:

    - an action whose ``action_id`` differs from the bound target abnormal
      move is not this hook's business: it returns UNCHANGED, with no
      readiness call and no mutation;
    - the target action must carry ``type == 'move_entity'`` AND
      ``entity_id == binding['receiver_owner']`` (the predeclared receiver).
      Wrong or missing type, or wrong or missing receiver, is an explicit
      identity/configuration error raised BEFORE any readiness call — never
      an ordinary pass-through and never an omission;
    - the correct target passes through unchanged when the real pure
      readiness check passes at the dispatcher's current instant
      ``tick * STEP_NS``; it returns ``None`` (the engine's
      explicit-omission disposition) and records the concrete refusal in
      ``refusals`` when the check refuses. A refused first execution never
      reaches the handler and is never called applied.
    """
    if type(tick) is not int or tick < 0:
        raise ValueError("tick must be the dispatcher's current nonnegative integer tick")
    check_fn = default_readiness_check if readiness_check is None else readiness_check
    if action.get("action_id") != binding["action"]:
        # Another event's action on the shared engine filter: pass through
        # UNCHANGED, with no readiness call and no mutation.
        return action
    # Target action identity: the bound abnormal move must carry the move
    # handler type AND the predeclared binding receiver before any readiness
    # call. Wrong or missing type, or wrong or missing receiver, is an
    # explicit identity/configuration error — never an ordinary pass-through.
    if action.get("type") != "move_entity":
        raise ValueError(
            f"target action {binding['action']!r} must have type 'move_entity', "
            f"got {action.get('type')!r}")
    if action.get("entity_id") != binding["receiver_owner"]:
        raise ValueError(
            f"prefix hook bound to ({binding['action']!r} -> {binding['receiver_owner']!r}) "
            f"received the target action for a different entity: "
            f"{action.get('entity_id')!r}")
    check = check_fn(binding, receipt_record, execution_ns=tick * A.STEP_NS,
                     movement_epoch0_active=movement_epoch0_active)
    if check["status"] != "ready_for_executor":
        if refusals is not None:
            refusals.append({
                "schema_version": A.SCHEMA, "hook": "single_action_prefix",
                "disposition": "first_execution_refused", "applied": False,
                "dispatch_tick": tick, "execution_ns": tick * A.STEP_NS,
                "reason": check["reason"], "readiness": copy.deepcopy(check),
            })
        return None
    return action


def single_action_decision_gate(event_def, tick, engine, interpreter, *,
                                binding, receipt_record,
                                movement_epoch0_active=True,
                                readiness_check=None):
    """Engine decision_gate: defer the target event until its receipt is live.

    Exact ``{'allow', 'evidence'}`` contract of the engine's admission
    callback. For the target event the check is the receiver-controller
    admission at the CURRENT tick (actual accepted native RX, identity join,
    active authority) followed by the same pure readiness check the dispatcher
    will repeat immediately before the handler. A refusal DEFERS the event to
    the next tick (never omits an action from here, never consumes
    ``max_fire_count``); every other event passes through untouched.
    """
    if type(tick) is not int or tick < 0:
        raise ValueError("tick must be the engine's current nonnegative integer tick")
    event_id = event_def.get("event_id")
    if event_id != TARGET_EVENT_ID:
        return {"allow": True,
                "evidence": {"gate": "single_action_prefix", "event_id": event_id,
                             "decision_tick": tick, "decision": "pass_through"}}
    observation_ns = tick * A.STEP_NS
    admission = A.admit_movement_command(binding, receipt_record, observation_ns,
                                         movement_epoch0_active=movement_epoch0_active)
    if admission["decision"] != "admitted":
        return {"allow": False,
                "evidence": {"gate": "single_action_prefix", "event_id": event_id,
                             "decision_tick": tick, "decision": "defer_no_admitted_receipt",
                             "reason": admission["reason"],
                             "gate_evidence": admission}}
    check_fn = default_readiness_check if readiness_check is None else readiness_check
    check = check_fn(binding, receipt_record, execution_ns=observation_ns,
                     movement_epoch0_active=movement_epoch0_active)
    if check["status"] != "ready_for_executor":
        return {"allow": False,
                "evidence": {"gate": "single_action_prefix", "event_id": event_id,
                             "decision_tick": tick, "decision": "defer_first_execution_refused",
                             "reason": check["reason"], "gate_evidence": admission,
                             "readiness": check}}
    return {"allow": True,
            "evidence": {"gate": "single_action_prefix", "event_id": event_id,
                         "decision_tick": tick, "decision": "admitted_ready",
                         "gate_evidence": admission, "readiness": check}}


def wire_single_action_prefix(scene, script, script_path, episode_id, seed, *,
                              binding, receipt_record, duration_ticks=None,
                              readiness_check=None, execute_linked=None):
    """Pass the hook into the EXISTING ``execute_linked`` call and run it.

    ``execute_linked`` defaults to the real
    ``Dataset.semantic_simulation.ns3_episode.linked_engine.execute_linked``.
    The decision gate and action filter are this function's own closures over the
    submitted binding and same-run receipt record — no future story content is
    read and no other callback is added. Every concrete first-execution refusal
    recorded by the dispatcher side is surfaced on the returned run dict under
    ``prefix_refusals``.
    """
    if execute_linked is None:
        from Dataset.semantic_simulation.ns3_episode.linked_engine import \
            execute_linked as execute_linked_fn
    else:
        execute_linked_fn = execute_linked
    refusals: list = []

    def decision_gate(event_def, tick, engine, interpreter):
        return single_action_decision_gate(
            event_def, tick, engine, interpreter, binding=binding,
            receipt_record=receipt_record, readiness_check=readiness_check)

    def action_filter(action, tick):
        return prefix_action(action, tick=tick, binding=binding,
                             receipt_record=receipt_record,
                             readiness_check=readiness_check,
                             refusals=refusals)

    kwargs = {} if duration_ticks is None else {"duration_ticks": duration_ticks}
    run = execute_linked_fn(scene, script, script_path, episode_id, seed=seed,
                            decision_gate=decision_gate,
                            action_filter=action_filter, **kwargs)
    run["prefix_refusals"] = refusals
    return run


RECOVERY_EVENT_ID = "secure_recovery"
LANDING_EVENT_ID = "lifecycle_landing_uav_digital_l6_5_v1"
GCS_SECURE_ACTION = "set_runtime_state_l6_5_v1_secure_recovery_00"
UAV_SECURE_ACTION = "set_runtime_state_l6_5_v1_secure_recovery_01"
GCS_LOCKOUT_ACTION = "set_runtime_state_l6_5_v1_command_lockout_00"


def wire_receipt_story(scene, script, script_path, episode_id, seed, *,
                       bindings, receipts, messages):
    """Execute R2 on the existing engine callbacks, including actual recovery.

    The authored UAV patch is held out of the engine queue at secure-event
    dispatch. Once the native status is available and original +5 is due on
    the next tick, observe_tick dispatches that same patch with delay 1.
    Pending patches apply before row recording on the following tick. Missing
    status leaves it unexecuted and blocks landing; omissions are not success.
    """
    from Dataset.semantic_simulation.ns3_episode.linked_engine import execute_linked

    actions = {a["action_id"]: a for e in script["events"] for a in e["actions"]}
    triggers = {t["trigger_id"]: t for t in script["triggers"]}
    events = {e["event_id"]: e for e in script["events"]}
    recovery_delay = actions[UAV_SECURE_ACTION]["delay_ticks"]
    landing_delay = triggers[events[LANDING_EVENT_ID]["trigger_ref"]]["delay_ticks"]
    state = {"movement_epoch0_active": True, "revoked_at_tick": None,
             "secure_event_tick": None, "gcs_secure_applied_tick": None,
             "status_send_tick": None, "landing_send_tick": None,
             "recovery_admission": None, "uav_recovery_due_tick": None,
             "uav_recovery_queue_tick": None, "uav_recovery_applied_tick": None,
             "landing_admission": None}
    observations, deferred, injected_audit = [], [], []
    pending = [None]

    def landing_check(tick):
        if (state["landing_send_tick"] is None or
                messages[A.LANDING_ACTION]["send_ns"] != state["landing_send_tick"] * A.STEP_NS):
            return {"status": "refused", "applied": False,
                    "reason": "landing_schedule_not_current", "execution_ns": tick * A.STEP_NS}
        return A.check_landing_execution_ready(
            bindings[A.LANDING_ACTION], receipts[A.LANDING_ACTION],
            execution_ns=tick * A.STEP_NS,
            recovery_admission=state["recovery_admission"],
            uav_recovery_applied=state["uav_recovery_applied_tick"] is not None,
            movement_epoch0_active=state["movement_epoch0_active"])

    def decision_gate(event_def, tick, engine, interpreter):
        if event_def["event_id"] == LANDING_EVENT_ID:
            check = landing_check(tick)
            if check["status"] == "ready_for_executor":
                state["landing_admission"] = copy.deepcopy(check)
            return {"allow": check["status"] == "ready_for_executor", "evidence": check}
        return single_action_decision_gate(event_def, tick, engine, interpreter,
            binding=bindings[A.MOVEMENT_ACTION], receipt_record=receipts[A.MOVEMENT_ACTION],
            movement_epoch0_active=state["movement_epoch0_active"])

    def action_filter(action, tick):
        aid = action["action_id"]
        if aid == GCS_SECURE_ACTION:
            state["secure_event_tick"] = tick
            state["uav_recovery_due_tick"] = tick + recovery_delay
            state["landing_send_tick"] = tick + landing_delay
        if aid == UAV_SECURE_ACTION:
            pending[0] = copy.deepcopy(action)
            deferred.append({"tick": tick, "action_id": aid, "applied": False,
                "disposition": "held_for_native_recovery_status",
                "authored_due_tick": tick + action["delay_ticks"]})
            return None
        if aid == A.LANDING_ACTION:
            check = landing_check(tick)
            if check["status"] != "ready_for_executor":
                raise RuntimeError("landing dispatch refused after gate: " + str(check))
            if action["entity_id"] != bindings[A.LANDING_ACTION]["receiver_owner"]:
                raise ValueError("landing action receiver differs from receipt binding")
            # The frozen script's tick-460 estimate describes its authored
            # plan. Preserve it as source metadata, never as a receipt-gated
            # dispatch feasibility proof. The engine consumes only the
            # landing reference from this field to classify landing motion.
            authored_estimate = action.pop("terminal_feasibility")
            action["authored_terminal_estimate"] = {
                "source_ref": str(script_path),
                "semantics": "historical authored plan estimate; not evidence of actual dispatch or terminal success",
                "estimate": authored_estimate,
            }
            action["terminal_feasibility"] = {
                "semantics": "landing geometry only; contains no dispatch feasibility proof",
                "landing_reference_enu_m": authored_estimate["landing_reference_enu_m"],
                "touchdown_altitude_tolerance_m": authored_estimate["touchdown_altitude_tolerance_m"],
                "touchdown_dwell_ticks": authored_estimate["touchdown_dwell_ticks"],
            }
            return action
        return prefix_action(action, tick=tick, binding=bindings[A.MOVEMENT_ACTION],
            receipt_record=receipts[A.MOVEMENT_ACTION],
            movement_epoch0_active=state["movement_epoch0_active"], refusals=deferred)

    def observe_tick(tick, engine, interpreter, rows):
        if state["movement_epoch0_active"]:
            for action in engine.executed_actions:
                if (action["action_id"] == GCS_LOCKOUT_ACTION
                        and action["result"]["effective_tick"] <= tick):
                    state["movement_epoch0_active"] = False
                    state["revoked_at_tick"] = tick
                    observations.append({"tick": tick, "kind": "movement_epoch0_revoked",
                                         "action": copy.deepcopy(action)})
                    break
        if state["secure_event_tick"] is None:
            return
        if state["gcs_secure_applied_tick"] is None:
            for action in engine.executed_actions:
                if (action["action_id"] == GCS_SECURE_ACTION
                        and action["result"]["effective_tick"] <= tick):
                    security = engine.entities[A.GCS]["security_state"]
                    if security["gcs_compromised"] or security["command_integrity_violation"]:
                        raise RuntimeError("GCS secure patch due but secure state not observed")
                    state["gcs_secure_applied_tick"] = tick
                    state["status_send_tick"] = tick
                    observations.append({"tick": tick, "kind": "secure_status_send_eligible",
                        "status": A.secure_status_record(state["secure_event_tick"], tick, tick),
                        "gcs_security_state": copy.deepcopy(security)})
                    break
        if state["gcs_secure_applied_tick"] is None:
            return
        if state["recovery_admission"] is None:
            # A coupling iteration carrying an earlier schedule cannot send
            # status before this run's actual secure patch. No backdating.
            sent = messages[A.NOTIFICATION_ACTION]["send_ns"] // A.STEP_NS
            if sent != state["status_send_tick"]:
                if tick == state["gcs_secure_applied_tick"]:
                    observations.append({"tick": tick, "kind": "notification_schedule_not_current",
                        "submitted_send_tick": sent, "actual_send_tick": state["status_send_tick"]})
                return
            admission = A.admit_recovery_notification(bindings[A.NOTIFICATION_ACTION],
                receipts[A.NOTIFICATION_ACTION], tick * A.STEP_NS,
                incident_movement_epoch=A.revoked_movement_epoch())
            if admission["decision"] == "admitted":
                state["recovery_admission"] = admission
                observations.append({"tick": tick, "kind": "recovery_knowledge_acquired",
                                     "admission": copy.deepcopy(admission)})
        if pending[0] is not None and state["recovery_admission"] is not None:
            if tick + 1 >= state["uav_recovery_due_tick"]:
                action = copy.deepcopy(pending[0])
                action["delay_ticks"] = 1
                result = engine._handle_set_runtime_state(action, tick)
                if result["status"] != "ok":
                    raise RuntimeError(result)
                state["uav_recovery_queue_tick"] = tick
                injected_audit.append({"tick": tick, "action": action, "result": result,
                    "origin": "authored recovery patch deferred until actual receipt",
                    "authored_delay_ticks": recovery_delay})
                observations.append({"tick": tick, "kind": "uav_recovery_queued",
                    "effective_tick": result["effective_tick"],
                    "authored_due_tick": state["uav_recovery_due_tick"]})
                pending[0] = None
        if state["uav_recovery_queue_tick"] is not None and state["uav_recovery_applied_tick"] is None:
            if tick > state["uav_recovery_queue_tick"]:
                navigation = engine.entities[A.UAV]["navigation_state"]
                security = engine.entities[A.UAV]["security_state"]
                if not navigation["route_recovered"] or security["command_lockout"]:
                    raise RuntimeError("UAV recovery queued but actual state not applied")
                state["uav_recovery_applied_tick"] = tick
                observations.append({"tick": tick, "kind": "uav_recovery_applied",
                    "navigation_state": copy.deepcopy(navigation),
                    "security_state": copy.deepcopy(security)})

    run = execute_linked(scene, script, script_path, episode_id, seed=seed,
        decision_gate=decision_gate, action_filter=action_filter, observe_tick=observe_tick)
    run["audit"].extend(injected_audit)
    run["audit"].sort(key=lambda row: row["tick"])
    run["prefix_refusals"] = deferred
    run["story_state"] = state
    run["recovery_observations"] = observations
    return run

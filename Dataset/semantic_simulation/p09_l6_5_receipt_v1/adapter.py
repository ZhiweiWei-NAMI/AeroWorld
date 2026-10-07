"""Minimal existing-interface adapter for the L6-5_v1 seed00 receipt story.

L6-5 native-directed patch v4 (pure execution-readiness check, this session),
on top of the v1 identity-join repair and the v2 explicit control epoch:
- ``bind_command`` requires an EXPLICIT ``control_epoch`` (no None/source-life
  fallback). The control epoch is verified against the PREDECLARED profile
  grants — production predeclares ONLY epoch0 — never self-authorized by the
  caller; the transport life epoch stays independent of it.
- The identity helper verifies ``binding["control_epoch"]`` against the
  binding's own predeclared profile grants (fixed 0 is gone).
- ``check_movement_execution_ready`` is a PURE CHECK, never an execution or a
  dispatch: at the AUTHORITATIVE EXECUTOR's required current ``execution_ns``
  (no default, no +1 ns, no backdating) it rechecks the real RX and the
  active authority, refuses ``execution_ns >= deadline_ns`` via
  ``execution_expiry_status``, and on success returns
  ``ready_for_executor`` with ``applied=False`` and the candidate
  ``execution_ns``. It never creates a physical/dispatch fact; real
  execution facts belong to a future, separately authorized executor
  integration. No historical/dictionary prior-dispatch API exists.
- Timely recovery-status KNOWLEDGE still persists beyond its delivery TTL.

v1 (previous session, retained): a movement/recovery admission joins the actual
frozen receipt to the PREDECLARED binding on the full submitted identity BEFORE
any time or revocation check; a missing or mismatched identity field is an
explicit rejection. The transport ``source_epoch`` (ns-3 endpoint lifetime
generation) is separate from the movement control-authority epoch; recovery
binds the revoked control epoch explicitly and never revives it.

Reuses only existing shared modules: the ns-3.48 timed-flow transport in
``Dataset.semantic_simulation.ns3_episode.provider_adapter`` and the receipt
contracts in ``linked_transport`` / ``command_receipts`` /
``controller_receipt_consumer``. This module adds no transport, no analytical
delivery, no new physical entity, no credential or attack tooling.

Approved representative contract (native brief, unchanged):
- The compromised GCS movement role stays the authorized control sender,
  epoch0, for the whole vulnerable phase. The original 9 m/s abnormal command
  must really TX (native ``first_tx``), be socket-accepted, be natively
  received, and be admitted by the receiver controller before the authored
  motion effect.
- The original 31-tick owner-local isolation revokes the movement epoch0
  authority and flies the original 4 m/s safe route; the lockout is reported
  from the actual revoked-epoch state.
- Secure recovery fires at the original applied time; the GCS delay-1 secure
  runtime patch must actually apply before any status send; status records the
  original secure event time AND the applied state time.
- UAV recovery effects wait for BOTH the original +5 due time and actual RX
  availability; landing waits for the original +80 due AND RX availability.
  Nothing is backdated.
- The same original GCS endpoint holds a separately predeclared recovery
  notification authority (synthetic assumption, declared below): the movement
  role is compromised, the notification signing path is independently trusted.
- Authority binds to flow/message then to the native receipt. A payload
  claiming a role never grants authority. Recovery binds the current incident /
  revoked epoch but never consumes or revives the revoked movement authority.

Caption corrections from the parent review of checkpoint 1 (applied to this
file and the local README; the frozen review packet keeps its original text):
- ``first_tx_ns`` is the native socket-acceptance time of an attempt; taken
  alone it is not proof of over-the-air PHY transmission.
- 27.0 s socket-acceptance -> 27.0023263 s RX is the 2.3263 ms native
  airtime/propagation segment. The remaining RX -> 27.5 s admission-barrier
  interval (497.6737 ms) is the barrier-minus-RX interval: only the two
  instants come from the frozen trace; the physical wait between them is
  unverified (the receiver-grid interpretation is an assumption, not proven).
- ``physical_motion_evidence: null`` in an admission record means "motion not
  evaluated by this gate"; it is never proof that the 9 m/s motion executed.
  Motion evidence is bound separately from the actual frozen trajectory rows by
  ``bind_motion_to_admission``, which is itself a causal gate: it fails closed
  unless the movement action actually executed at or after the admission
  barrier and the rows show real displacement toward the commanded targets.
"""
from __future__ import annotations

import copy
import sys

from Dataset.semantic_simulation.ns3_episode.command_receipts import (
    CommandReception,
    CommandTransmission,
    controller_receipts_before,
    record_downlink_receipt,
)
from Dataset.semantic_simulation.ns3_episode.linked_transport import (
    message_flow,
    message_receipts,
    owner_node,
)

SCHEMA = "p09.l6_5.command_receipt_authority/v1"
STEP_NS = 100_000_000
EPISODE_ID = "L6-5_v1__seed00"
SCENARIO_ID = "L6-5_v1"
GCS = "gcs_anchor_l6_5_v1"
UAV = "uav_digital_l6_5_v1"


def runtime_identity():
    """Interpreter identity for receipts: the native-approved airfogsim python.

    Native requires every project diagnostic/inline script to run under
    ``/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python``.
    No package installation and no environment alteration; a non-conforming
    interpreter is reported, not silently accepted.
    """
    approved = "/home/weizhiwei/data/iiot_predict/iiot_py311/airfogsim/bin/python"
    executable = sys.executable
    return {"approved_interpreter": approved, "sys_executable": executable,
            "matches_approved": executable == approved}

# Authored transport parameters, shared verbatim with the existing P09
# ``linked_contract`` command profile (payload 64 B, 3 attempts, 100 ms period,
# 2 s deadline, zero policy delay). These are declared simulation assumptions,
# not measured hardware behaviour.
COMMAND = {"payload_bytes": 64, "attempts": 3, "period_ns": STEP_NS,
           "deadline_ns": 2_000_000_000, "policy_delay_ns": 0}

# Original source story ticks (interpreter event grid: events fire on the
# authored 5-tick grid, so delayed events land on the next grid tick).
#
# ``first_command_admission_barrier_tick`` (previously named
# ``uav_recovery_due_tick``) is the receiver-side first-command OBSERVATION
# barrier of the checkpoint-1 record: 27.5 s. It is the instant at which the
# receiver controller is observed to evaluate the first movement command. It is
# NOT an executor slot, NOT a due time of the UAV recovery effect, and it adds
# no new physical evidence. The actual recovery effect is authored as
# secure_event + 5 with an actually available received status (tick 385 from
# effective_tick), and landing as secure_event + 80 (tick 460), each gated by
# RX availability. The physical wait between the RX instant and this barrier
# (barrier-minus-RX interval) is unverified; only the RX instant and the
# barrier instant themselves come from the frozen trace.
STORY = {
    "takeoff_event_tick": 30,
    "gcs_intrusion_event_tick": 240,
    "abnormal_command_event_tick": 270,      # vulnerable-phase command
    "command_lockout_event_tick": 305,       # 270 + 31, next 5-tick grid tick
    "secure_recovery_event_tick": 380,       # 305 + 71, next 5-tick grid tick
    "landing_event_tick": 460,               # 380 + 80, next 5-tick grid tick
    "first_command_admission_barrier_tick": 275,  # abnormal event + 5: first-command OBSERVATION barrier only
    # Legacy key kept because read-only callers index it (command_trace.py:41
    # ADMISSION_TICK, coupled_episode.py:69 ABNORMAL_ADMISSION_TICK). Its
    # meaning is the same observation barrier above, NOT a recovery due time;
    # renaming it away would break those readers, so both names denote 275 and
    # the recovery effect uses secure_event + 5 = 385 instead.
    "uav_recovery_due_tick": 275,
    "recovery_due_offset_ticks": 5,          # actual recovery: secure_event + 5 with available received status
    "gcs_status_applied_tick": 381,          # secure event + authored delay 1
}

MOVEMENT_ROLE = "gcs_uav_movement_control"
NOTIFICATION_ROLE = "gcs_recovery_notification"
MOVEMENT_ACTION = "move_uav_intrusion_abnormal"
NOTIFICATION_ACTION = "gcs_secure_recovery_notification"
LANDING_ACTION = "move_uav_digital_l6_5_v1_landing_return"

# Predeclared authority PROFILES: the production default predeclares ONLY the
# frozen seed-00 story's epoch0 control authority (movement epoch0 control
# phase, revoked by the original 31-tick owner isolation; story notification
# authority at epoch0). No other control epoch is predeclared;
# no production authorization mechanism exists beyond this table.
# Control epochs are NOT fixed by the transport life and are NOT inferred from
# it; callers must present the predeclared epoch explicitly and cannot
# self-authorize another.
AUTHORITY_PROFILES = {
    0: {"control_epoch": 0,
        "grants": {
            (MOVEMENT_ROLE, MOVEMENT_ACTION): (GCS, 0),   # epoch0 control phase
            (NOTIFICATION_ROLE, NOTIFICATION_ACTION): (GCS, 0),
            (MOVEMENT_ROLE, LANDING_ACTION): (GCS, 0),    # original landing command
        }},
}
PREDECLARED_CONTROL_EPOCHS = (0,)

# Separation of the two epoch namespaces:
# - TRANSPORT source_epoch = the ns-3 endpoint LIFETIME generation
#   (``linked_transport.flow`` ``life_epoch``); it is an explicit transport
#   fact and is INDEPENDENT of the control epoch: the same control authority
#   may legally be presented over a newer endpoint lifetime (e.g. life 7 with
#   control 0), and a re-connected endpoint is a new lifetime.
# - The movement CONTROL-AUTHORITY epoch is the predeclared per-profile grant
#   above; profile 0's epoch0 is revoked by the original 31-tick owner
#   isolation and never rearmed.
# A recovery notification may arrive from a NEWER transport lifetime while
# binding a predeclared control/incident epoch; the two namespaces are joined
# explicitly, never conflated.


def revoked_movement_epoch() -> int:
    """The predeclared movement control epoch the story isolation revokes."""
    return AUTHORITY_PROFILES[0]["control_epoch"]


def execution_expiry_status(execution_ns, deadline_ns):
    """Mirror of the shared consumer expiry rule for a single planned execution.

    The shared ``consume_received_commands`` marks a receipt
    ``expired_before_execution`` when the planned execution instant is at or
    after the receipt's transport deadline, else ``ready_for_executor``. This
    helper exposes that same rule without mutating the shared module: first
    movement execution at ``execution_ns >= deadline_ns`` is expired, strictly
    before it is ready. Boundary: exactly at the deadline (29 s for the frozen
    movement receipt) is already expired.
    """
    if type(execution_ns) is not int or execution_ns < 0:
        raise ValueError("execution_ns must be a nonnegative integer nanosecond instant")
    if type(deadline_ns) is not int or deadline_ns < 0:
        raise ValueError("deadline_ns must be a nonnegative integer nanosecond instant")
    return "expired_before_execution" if execution_ns >= deadline_ns else "ready_for_executor"


def bind_incident_control_epoch(*, incident_movement_epoch, transport_source_epoch):
    """Explicit incident binding: predeclared control epoch vs transport lifetime.

    This is the smallest control/incident binding needed by the gates. It
    carries BOTH namespaces separately so the notification gate can require a
    receipt from the exact transport lifetime (endpoint generation) while
    binding the declared movement control epoch as the reported incident. The
    control epoch must be PREDECLARED (``PREDECLARED_CONTROL_EPOCHS``): a
    caller cannot self-authorize an incident epoch by passing it. The two may
    legally differ (a recovery sent over a newer endpoint lifetime binds an
    older predeclared control epoch); neither defaults to the other and a
    missing value is an error.
    """
    if type(incident_movement_epoch) is not int or incident_movement_epoch < 0:
        raise ValueError(f"incident_movement_epoch must be an explicit nonnegative integer epoch, got {incident_movement_epoch!r}")
    if incident_movement_epoch not in PREDECLARED_CONTROL_EPOCHS:
        raise ValueError(
            f"incident_movement_epoch {incident_movement_epoch!r} is not a predeclared control epoch; "
            f"predeclared: {list(PREDECLARED_CONTROL_EPOCHS)}")
    if type(transport_source_epoch) is not int or transport_source_epoch < 0:
        raise ValueError(f"transport_source_epoch must be an explicit nonnegative integer epoch, got {transport_source_epoch!r}")
    return {"incident_movement_epoch": incident_movement_epoch,
            "transport_source_epoch": transport_source_epoch,
            "predeclared_control_epoch": AUTHORITY_PROFILES[incident_movement_epoch]["control_epoch"],
            "epochs_differ": incident_movement_epoch != transport_source_epoch,
            "transport_epoch_is_lifetime_generation": True,
            "control_epoch_is_revocable_movement_authority": True}


def bind_command(message_id, action, declared_role, payload, *, source_owner,
                 source_epoch, receiver_owner, receiver_epoch, control_epoch):
    """Bind authority -> message before any transport exists.

    The payload is stored verbatim for the trace but is never consulted for
    authority: a payload claiming a role grants nothing. ``control_epoch`` is
    EXPLICIT REQUIRED (native-directed v2): there is no None and no source-life
    fallback. It must be one of the PREDECLARED control epochs
    (``PREDECLARED_CONTROL_EPOCHS`` / ``AUTHORITY_PROFILES``) — a caller cannot
    self-authorize a control epoch by passing it. ``source_epoch`` is the
    TRANSPORT lifetime epoch of the sender endpoint (ns-3 endpoint generation);
    it is an independent transport fact and is NOT matched against the control
    epoch.
    """
    if type(control_epoch) is not int or control_epoch < 0:
        raise ValueError(f"control_epoch must be an explicit nonnegative integer, got {control_epoch!r}")
    profile = AUTHORITY_PROFILES.get(control_epoch)
    if profile is None:
        raise ValueError(
            f"control_epoch {control_epoch!r} is not a predeclared authority profile; "
            f"predeclared control epochs: {list(AUTHORITY_PROFILES)}")
    grant = profile["grants"].get((declared_role, action))
    if grant is None:
        raise ValueError(f"no predeclared authority: role={declared_role!r} action={action!r}")
    if (source_owner, control_epoch) != grant:
        raise ValueError(
            f"declared authority does not match the predeclared sender identity: "
            f"role={declared_role!r} sender=({source_owner!r}, control={control_epoch}) grant={grant}")
    return {
        "schema_version": SCHEMA, "message_id": message_id, "action": action,
        "declared_role": declared_role, "source_owner": source_owner,
        "source_epoch": source_epoch, "control_epoch": control_epoch,
        "control_epoch_profile": AUTHORITY_PROFILES[control_epoch],
        "receiver_owner": receiver_owner,
        "receiver_epoch": receiver_epoch, "payload_claimed_roles": sorted(
            payload["claimed_roles"]) if isinstance(payload, dict) and "claimed_roles" in payload else [],
        "authority_source": "predeclared adapter grant table; payload claims never grant authority",
    }


def build_message(episode, binding, send_tick, evidence_ns):
    """Wrap the bound message with the existing transport window contract."""
    if type(send_tick) is not int or send_tick < 0:
        raise ValueError("command send tick must be a nonnegative integer original tick")
    message = message_flow(
        episode, {"command": COMMAND, "pad_request_transport": COMMAND}, binding["message_id"],
        binding["source_owner"], binding["receiver_owner"], send_tick * STEP_NS,
        evidence_ns, "flight_command")  # any non-'request_pad' action selects the command profile
    message["action"] = binding["action"]
    message["binding"] = binding
    return message


def extract_receipts(episode, message, network):
    """Native per-attempt receipts through the existing shared contract."""
    receipts = message_receipts(episode, [message], network)
    record = receipts[message["message_id"]]
    record["receiver_epoch_expected"] = message["binding"]["receiver_epoch"]
    return record


def _receipt_identity_rejected(binding, accepted, *, require_source_epoch):
    """Join the actual receipt to the PREDECLARED binding on full identity.

    Runs BEFORE any time or revocation check. Every field must equal the
    binding exactly; any missing or mismatched field is an explicit rejection
    reason (never repaired with an inferred default). Checked fields:

    - ``command_id`` == binding message_id
    - ``action`` == binding action
    - ``source_owner`` == binding source_owner
    - ``receiver_owner`` == binding receiver_owner
    - ``receiver_epoch`` == binding receiver_epoch (transport lifetime of RX)
    - ``source_epoch`` == binding source_epoch (transport lifetime of the
      sender: the ns-3 endpoint generation, NOT the movement control epoch)
    - the declared role must hold the predeclared grant for this action from
      this sender and must not be claimed from the payload

    ``require_source_epoch`` is the transport source lifetime epoch the
    receipt must carry. The CONTROL authority epoch is NOT a fixed constant
    and is never self-authorized by the caller: it is read from the submitted
    PREDECLARED binding (``binding["control_epoch"]``), which ``bind_command``
    only issues for predeclared control epochs, and the grant check uses that
    binding's own profile table.
    """
    if type(accepted) is not dict:
        return "receipt_accepted_row_missing_or_malformed"
    if not (require_source_epoch == int(require_source_epoch)):
        return "binding_source_epoch_not_an_integer"
    required = {
        "command_id": binding["message_id"],
        "action": binding["action"],
        "source_owner": binding["source_owner"],
        "receiver_owner": binding["receiver_owner"],
        "receiver_epoch": binding["receiver_epoch"],
        "source_epoch": require_source_epoch,
    }
    reasons = {
        "command_id": "receipt_command_id_differs_from_binding",
        "action": "receipt_action_differs_from_binding",
        "source_owner": "receipt_source_owner_differs_from_binding",
        "receiver_owner": "receipt_receiver_owner_differs_from_binding",
        "receiver_epoch": "receipt_receiver_epoch_differs_from_binding",
        "source_epoch": "receipt_source_lifetime_epoch_differs_from_binding",
    }
    for field, expected in required.items():
        if field not in accepted:
            return f"receipt_missing_{field}"
        if accepted[field] != expected:
            return reasons[field]
    # Role authority: from the PREDECLARED grant table of the binding's own
    # control-epoch profile. The receipt/payload carries no role and never
    # carries the control epoch; the binding's control epoch is the one
    # ``bind_command`` predeclared (checked here so a receipt for a binding
    # whose declared role lost its grant is rejected too).
    profile = AUTHORITY_PROFILES.get(binding["control_epoch"])
    if profile is None:
        return "binding_control_epoch_not_predeclared"
    grant = profile["grants"].get((binding["declared_role"], binding["action"]))
    if grant is None or grant != (binding["source_owner"], binding["control_epoch"]):
        return "binding_declared_role_has_no_predeclared_grant_for_this_sender"
    return None


def admit_movement_command(binding, receipt_record, observation_ns, *, movement_epoch0_active):
    """Receiver-controller admission gate for a movement command.

    Order of checks (this repair):
    1. Receipt -> PREDECLARED binding identity join (exact command/message ID,
       action, source/receiver owner, source/receiver transport life epochs,
       declared role from the grant table). Missing or mismatched identity is
       an explicit rejection before any time or revocation check.
    2. Movement control authority still active (the predeclared epoch0 grant,
       revoked by the original isolation; never rearmed by recovery).
    3. Actual shared-contract availability: an accepted native RX strictly
       before the observation, through the existing
       ``controller_receipts_before`` semantics (unchanged shared module).
    A revoked epoch, a late or missing RX, or any identity mismatch never
    admits and never produces an early effect.
    """
    accepted = receipt_record.get("accepted")
    evidence = {
        "schema_version": SCHEMA, "gate": "movement_command_admission",
        "message_id": binding["message_id"], "declared_role": binding["declared_role"],
        "action": binding["action"], "movement_epoch0_active": movement_epoch0_active,
        "movement_control_authority_epoch": binding["control_epoch"],
        "movement_control_epoch_is_predeclared": True,
        "transport_source_epoch": binding["source_epoch"],
        "identity_join_checked": True,
        "observation_ns": observation_ns, "accepted": None, "decision": "rejected",
        "reason": None, "physical_motion_evidence": None,
    }
    # 1. identity join before time/revocation (the receipt is carried in the
    #    evidence even when the join fails, so the native review sees exactly
    #    which identity was rejected).
    if accepted is not None:
        identity_reason = _receipt_identity_rejected(
            binding, accepted, require_source_epoch=binding["source_epoch"])
        if identity_reason is not None:
            evidence["reason"] = identity_reason
            evidence["accepted"] = copy.deepcopy(accepted)
            return evidence
        evidence["accepted"] = copy.deepcopy(accepted)
    # 2. movement control authority (revocation): the binding's predeclared
    #    control epoch is the one the story isolation revokes; recovery never
    #    rearms it.
    if not movement_epoch0_active:
        evidence["reason"] = "movement_epoch0_authority_revoked"
        return evidence
    # 3. actual shared-contract availability
    if accepted is None:
        evidence["reason"] = "no_accepted_native_rx"
        evidence["attempts"] = copy.deepcopy(receipt_record.get("attempts", []))
        return evidence
    ready = controller_receipts_before(
        [accepted], observation_ns, receiver_owner=binding["receiver_owner"],
        receiver_epoch=binding["receiver_epoch"])
    if not ready:
        evidence["reason"] = "receipt_not_available_strictly_before_observation"
        return evidence
    evidence["decision"] = "admitted"
    evidence["admission_ns"] = observation_ns
    return evidence


def admit_recovery_notification(binding, receipt_record, observation_ns, *,
                                incident_movement_epoch):
    """Notification-path admission: trusted signing path, movement stays revoked.

    Order of checks (this repair):
    1. The declared authority must be the notification role (a movement
       binding can never be admitted here, so a movement receipt can never
       become a recovery receipt).
    2. Receipt -> PREDECLARED binding identity join on the full submitted
       identity (exact command/message ID, action, owners, transport life
       epochs) before any time check.
    3. Explicit control/incident binding, with the two epoch namespaces kept
       separate:
       - the receipt's transport ``source_epoch`` (endpoint lifetime) must
         equal the binding's transport source lifetime epoch;
       - the submitted binding must bind the current incident, i.e. the
         revoked movement control epoch passed as ``incident_movement_epoch``.
       A recovery notification may legally arrive from a transport lifetime
       different from the revoked control epoch (that separation is the point
       of this repair); a wrong lifetime or a wrong incident epoch is rejected
       explicitly, each independently.
    4. Availability through the existing ``controller_receipts_before``
       semantics (unchanged shared module).

    Knowledge semantics: a timely accepted recovery status is ACQUIRED
    KNOWLEDGE at its acceptance instant. The status row's transport delivery
    ``deadline_ns`` bounds delivery only; it does not expire that knowledge
    (the landing gate at secure_event + 80 still uses it). A late or missing
    status never admits. The gate never consumes, re-grants or revives the
    revoked movement authority; the returned record states that explicitly.
    """
    if binding["declared_role"] != NOTIFICATION_ROLE:
        raise ValueError("recovery admission requires the declared notification authority")
    accepted = receipt_record.get("accepted")
    evidence = {
        "schema_version": SCHEMA, "gate": "recovery_notification_admission",
        "message_id": binding["message_id"], "declared_role": binding["declared_role"],
        "action": binding["action"], "incident_movement_epoch": incident_movement_epoch,
        "transport_source_epoch": binding["source_epoch"],
        "movement_epoch0_active_after_recovery": False,
        "revives_revoked_movement_authority": False,
        "identity_join_checked": True,
        "acquired_knowledge": {
            "timely_accepted_status_is_knowledge": True,
            "delivery_deadline_ns": None,
            "knowledge_expires_with_delivery_deadline": False,
            "late_or_missing_status_admits": False,
        },
        "observation_ns": observation_ns, "accepted": None, "decision": "rejected",
        "reason": None, "physical_motion_evidence": None,
    }
    # 2. identity join before any time check. A movement receipt can never
    #    become a recovery receipt: the role guard above rejects any movement
    #    binding, and the join below rejects a receipt whose command/action
    #    identity is the movement command.
    if accepted is not None:
        identity_reason = _receipt_identity_rejected(
            binding, accepted, require_source_epoch=binding["source_epoch"])
        if identity_reason is not None:
            evidence["reason"] = identity_reason
            evidence["accepted"] = copy.deepcopy(accepted)
            return evidence
        evidence["accepted"] = copy.deepcopy(accepted)
        evidence["acquired_knowledge"]["delivery_deadline_ns"] = accepted.get("deadline_ns")
    # 3a. the submitted binding must bind the CURRENT incident (revoked
    #     movement control epoch). The receipt never carries the control
    #     epoch; the binding declares it. An incident epoch that is not
    #     predeclared is an UNKNOWN incident: it is refused here with the
    #     existing rejection semantics (decision=rejected), so a receipt
    #     admission never aborts an episode on an unknown-incident
    #     configuration; the config helper may still raise for a direct
    #     configuration bind.
    if binding.get("incident_movement_epoch") != incident_movement_epoch:
        evidence["reason"] = "notification_binding_not_bound_to_current_incident_epoch"
        return evidence
    if incident_movement_epoch not in PREDECLARED_CONTROL_EPOCHS:
        evidence["reason"] = "notification_incident_control_epoch_unknown"
        return evidence
    # 3b. explicit control/incident binding record (control epoch vs transport
    #     lifetime kept separate; a newer lifetime is legal, a wrong one is not)
    incident = bind_incident_control_epoch(
        incident_movement_epoch=incident_movement_epoch,
        transport_source_epoch=binding["source_epoch"])
    evidence["incident_control_binding"] = incident
    if incident["incident_movement_epoch"] != revoked_movement_epoch():
        evidence["reason"] = "notification_not_bound_to_current_incident_epoch"
        return evidence
    # 4. availability (actual accepted RX strictly before the observation)
    if accepted is None:
        evidence["reason"] = "no_accepted_native_rx"
        return evidence
    ready = controller_receipts_before(
        [accepted], observation_ns, receiver_owner=binding["receiver_owner"],
        receiver_epoch=binding["receiver_epoch"])
    if not ready:
        evidence["reason"] = "receipt_not_available_strictly_before_observation"
        return evidence
    evidence["decision"] = "admitted"
    evidence["admission_ns"] = observation_ns
    return evidence


def check_movement_execution_ready(binding, receipt_record, *, execution_ns,
                                   movement_epoch0_active):
    """PURE execution-readiness check at the executor's current instant.

    This is a CHECK, never an execution or a dispatch: it creates no physical,
    motion or dispatch fact, always returns ``applied=False`` and leaves
    ``physical_motion_evidence`` absent (None). The shared
    ``consume_received_commands`` plans an execution; this helper re-checks,
    at the authoritative executor's CURRENT execution instant, that the
    planned move may still go ahead. Real execution facts belong to the actual
    executor's results/logs in a future, separately authorized integration —
    they are not produced here.

    Requirements (all explicit, no default, no ``+1 ns``, no backdating):
    1. ``execution_ns`` is REQUIRED: the authoritative executor's current
       execution instant as an explicit nonnegative integer; a missing or
       malformed value is an error, never a default.
    2. Six identity joins on the real accepted native RX (exact command id,
       action, source/receiver owner, source/receiver transport life epochs,
       predeclared role grant at the binding's explicit control epoch), then
    3. recheck at that instant: the accepted RX must be available STRICTLY
       BEFORE ``execution_ns`` (unchanged ``controller_receipts_before``
       semantics) and the movement control authority must still be ACTIVE
       (the authoritative authority state; a revoked epoch never moves).
    4. ``execution_expiry_status(execution_ns, deadline_ns)``: an execution
       instant at or after the transport deadline is
       ``expired_before_execution`` and the check refuses. A late query of an
       old accepted RX at 29 s or later as a NEW execution is refused; it is
       never converted into a dispatch or a completed move.

    On success the result is ``status="ready_for_executor"`` with
    ``applied=False`` and the candidate ``execution_ns``. Any refused case
    returns ``status="refused"`` with the explicit reason. No accepted row,
    inactive authority or wrong transport life refuses even if a caller
    invents prior history: this check reads only the submitted receipt
    record and the passed authority state.
    """
    if type(execution_ns) is not int or execution_ns < 0:
        raise ValueError(
            "execution_ns must be the authoritative executor's current execution "
            "instant: an explicit nonnegative integer nanosecond value with no default")
    if binding["declared_role"] != MOVEMENT_ROLE:
        raise ValueError("movement execution readiness requires the declared movement authority")
    accepted = receipt_record.get("accepted")
    check = {
        "schema_version": SCHEMA, "check": "movement_execution_ready",
        "message_id": binding["message_id"], "declared_role": binding["declared_role"],
        "action": binding["action"],
        "movement_control_epoch": binding["control_epoch"],
        "transport_source_epoch": binding["source_epoch"],
        "executor_authoritative_execution_ns": execution_ns,
        "movement_epoch0_active": movement_epoch0_active,
        "accepted": None, "status": "refused", "applied": False,
        "execution_status": None, "candidate_execution_ns": None,
        "reason": None, "physical_motion_evidence": None,
    }
    if accepted is not None:
        identity_reason = _receipt_identity_rejected(
            binding, accepted, require_source_epoch=binding["source_epoch"])
        if identity_reason is not None:
            check["reason"] = identity_reason
            check["accepted"] = copy.deepcopy(accepted)
            return check
        check["accepted"] = copy.deepcopy(accepted)
    # Recheck at the executor's current instant: real native RX strictly
    # before it (unchanged shared semantics).
    if accepted is None:
        check["reason"] = "no_accepted_native_rx_before_execution"
        return check
    ready = controller_receipts_before(
        [accepted], execution_ns, receiver_owner=binding["receiver_owner"],
        receiver_epoch=binding["receiver_epoch"])
    if not ready:
        check["reason"] = "receipt_not_available_strictly_before_execution"
        return check
    # The movement control authority must still be active at that instant.
    if not movement_epoch0_active:
        check["reason"] = "movement_epoch0_authority_revoked_at_execution"
        return check
    # Expiry at that instant (>= transport deadline refuses; a late query of
    # an old accepted RX as a NEW execution is refused, never backdated).
    deadline_ns = accepted.get("deadline_ns")
    expiry = execution_expiry_status(execution_ns, deadline_ns)
    if expiry != "ready_for_executor":
        check["reason"] = expiry
        return check
    check["status"] = "ready_for_executor"
    check["execution_status"] = "ready_for_executor"
    check["candidate_execution_ns"] = execution_ns
    return check
def uav_recovery_effect(recovery_evidence, attempts_status, due_tick, observation_tick):
    """UAV recovery effect waits for the original +5 due AND actual RX.

    Both conditions must hold at the observation. Nothing is backdated to the
    authored due time or to the sender decision time.
    """
    if recovery_evidence is None or recovery_evidence.get("decision") != "admitted":
        return {"schema_version": SCHEMA, "effect": "uav_recovery_route_state",
                "applied": False, "reason": "recovery_notification_not_admitted",
                "due_tick": due_tick, "observation_tick": observation_tick,
                "physical_motion_evidence": None}
    if observation_tick < due_tick:
        return {"schema_version": SCHEMA, "effect": "uav_recovery_route_state",
                "applied": False, "reason": "before_authored_due_tick",
                "due_tick": due_tick, "observation_tick": observation_tick,
                "physical_motion_evidence": None}
    if not attempts_status:
        return {"schema_version": SCHEMA, "effect": "uav_recovery_route_state",
                "applied": False, "reason": "rx_availability_not_established",
                "due_tick": due_tick, "observation_tick": observation_tick,
                "physical_motion_evidence": None}
    return {"schema_version": SCHEMA, "effect": "uav_recovery_route_state",
            "applied": True, "reason": None, "due_tick": due_tick,
            "observation_tick": observation_tick, "rx_available": True,
            "applied_tick": observation_tick, "physical_motion_evidence": None}


def landing_effect(recovery_evidence, landing_receipt_available, due_tick, observation_tick):
    """Landing waits for the original +80 due AND RX availability, never backdate."""
    if recovery_evidence is None or recovery_evidence.get("decision") != "admitted":
        return {"schema_version": SCHEMA, "effect": "uav_landing_return",
                "applied": False, "reason": "recovery_notification_not_admitted",
                "due_tick": due_tick, "observation_tick": observation_tick,
                "physical_motion_evidence": None}
    if observation_tick < due_tick:
        return {"schema_version": SCHEMA, "effect": "uav_landing_return",
                "applied": False, "reason": "before_authored_due_tick",
                "due_tick": due_tick, "observation_tick": observation_tick,
                "physical_motion_evidence": None}
    if not landing_receipt_available:
        return {"schema_version": SCHEMA, "effect": "uav_landing_return",
                "applied": False, "reason": "rx_availability_not_established",
                "due_tick": due_tick, "observation_tick": observation_tick,
                "physical_motion_evidence": None}
    return {"schema_version": SCHEMA, "effect": "uav_landing_return",
            "applied": True, "reason": None, "due_tick": due_tick,
            "observation_tick": observation_tick, "rx_available": True,
            "applied_tick": observation_tick, "physical_motion_evidence": None}


def check_landing_execution_ready(binding, receipt_record, *, execution_ns,
                                  recovery_admission, uav_recovery_applied,
                                  movement_epoch0_active):
    """Check the existing action-specific landing grant without rearming epoch0.

    The abnormal-control grant remains revoked. The separately predeclared
    grant for LANDING_ACTION is usable only after real acquired recovery
    knowledge and the UAV's applied recovery state, with a matching live RX.
    This record is a check; the handler and trajectories establish execution.
    """
    if type(execution_ns) is not int or execution_ns < 0:
        raise ValueError("landing execution_ns must be the real executor clock")
    if binding["action"] != LANDING_ACTION or binding["declared_role"] != MOVEMENT_ROLE:
        raise ValueError("landing requires the existing action-specific landing grant")
    accepted = receipt_record["accepted"]
    evidence = {"gate": "landing_execution_ready", "message_id": binding["message_id"],
                "action": binding["action"], "execution_ns": execution_ns,
                "status": "refused", "applied": False, "reason": None,
                "movement_epoch0_active": movement_epoch0_active,
                "revives_revoked_movement_authority": False,
                "authority_scope": "existing action-specific landing grant",
                "accepted": copy.deepcopy(accepted)}
    if accepted is None:
        evidence["reason"] = "no_accepted_native_landing_rx"
        return evidence
    reason = _receipt_identity_rejected(binding, accepted,
                                        require_source_epoch=binding["source_epoch"])
    if reason is not None:
        evidence["reason"] = reason
        return evidence
    if movement_epoch0_active:
        evidence["reason"] = "incident_abnormal_control_not_revoked"
        return evidence
    if recovery_admission is None or recovery_admission["decision"] != "admitted":
        evidence["reason"] = "recovery_knowledge_not_acquired"
        return evidence
    if not uav_recovery_applied:
        evidence["reason"] = "uav_recovery_state_not_applied"
        return evidence
    if not controller_receipts_before([accepted], execution_ns,
            receiver_owner=binding["receiver_owner"], receiver_epoch=binding["receiver_epoch"]):
        evidence["reason"] = "landing_receipt_not_available_strictly_before_execution"
        return evidence
    expiry = execution_expiry_status(execution_ns, accepted["deadline_ns"])
    if expiry != "ready_for_executor":
        evidence["reason"] = expiry
        return evidence
    evidence["status"] = "ready_for_executor"
    return evidence


def same_tick_effect_order(status_apply_tick, status_send_tick):
    """The GCS secure runtime patch must apply before any status send.

    Equality is allowed (send strictly after apply within the same tick is the
    only same-tick arrangement); a send earlier than the applied state is a
    contract violation.
    """
    if status_send_tick < status_apply_tick:
        raise ValueError(
            f"status send tick {status_send_tick} precedes the applied secure "
            f"state tick {status_apply_tick}")
    return {"status_apply_tick": status_apply_tick, "status_send_tick": status_send_tick,
            "ordered": True, "applied_state_precedes_send": status_apply_tick <= status_send_tick}


def secure_status_record(secure_event_tick, status_apply_tick, status_send_tick):
    """Status carries the original secure event time AND the applied state time."""
    same_tick_effect_order(status_apply_tick, status_send_tick)
    if status_apply_tick <= secure_event_tick:
        raise ValueError("secure runtime patch must apply strictly after the secure event")
    return {"schema_version": SCHEMA, "record": "gcs_secure_status",
            "secure_event_tick": secure_event_tick, "status_applied_tick": status_apply_tick,
            "status_sent_tick": status_send_tick}


# ---------------------------------------------------------------------------
# Coupled-phase additions (checkpoint 2): lockout, report, recovery, landing.
# All values below come from actual runs of the immutable original sources
# (Dataset/scenarios/L6_digital_layer/failure/L6-5_v1/) under the approved
# interpreter; nothing is backdated or assumed.

def movement_authority_state(engine, tick):
    """Actual movement-authority state from the real engine patch schedule.

    Reads the runtime-state patch schedule of the executed original script:
    epoch0 stays active until the original 31-tick owner-local isolation patch
    actually applies on the GCS (effective_tick 306 for event tick 305). The
    movement authority is never rearmed afterwards, because the original script
    contains no movement re-grant patch.
    """
    if type(tick) is not int or tick < 0:
        raise ValueError("authority tick must be a nonnegative integer original tick")
    applied = {}
    lockout_action_ids = []
    for action in engine.executed_actions:
        if action.get("type") != "set_runtime_state":
            continue
        entity_id = action.get("entity_id")
        aid = str(action.get("action_id"))
        if entity_id != GCS:
            continue
        result = action.get("result") or {}
        effective = result.get("effective_tick")
        if type(effective) is not int:
            raise ValueError(f"runtime-state action {aid!r} has no actual effective_tick")
        # executed_actions rows carry action_id/result/tick/type only; the
        # isolation patch is identified by its actual original action_id
        # (set_runtime_state_l6_5_v1_command_lockout_00 -> command_lockout true).
        is_lockout = aid.endswith("command_lockout_00")
        if is_lockout:
            lockout_action_ids.append(aid)
        applied.setdefault(effective, []).append({"command_lockout": is_lockout,
                                                  "action_id": aid})
    if not lockout_action_ids:
        raise ValueError("original command_lockout_00 patch not found among executed GCS actions")
    active = True  # predeclared epoch0 grant is the initial authority state
    revoked_at = None
    for effective in sorted(applied):
        if effective > tick:
            break
        for patch in applied[effective]:
            if patch["command_lockout"]:
                active = False
                revoked_at = effective
    return {"movement_epoch0_active": active, "revoked_at_effective_tick": revoked_at,
            "evaluated_tick": tick, "lockout_action_ids": lockout_action_ids,
            "gcs_patch_effective_ticks": sorted(applied)}


def lockout_report(engine, tick):
    """Actual lockout report from the real revoked-epoch state.

    Reads the actual executed actions: the owner-local isolation event
    (original delay 31) produced both the GCS lockout patch (effective_tick
    306) and the UAV lockout patch (effective_tick 310). The report exists only
    once both patches have actually applied and states ``revoked_epoch``,
    which is the movement epoch the isolation revoked.
    """
    gcs_lockout = uav_lockout = None
    revoke_action_tick = None
    for action in engine.executed_actions:
        aid = str(action.get("action_id"))
        result = action.get("result") or {}
        effective = result.get("effective_tick")
        if aid.endswith("command_lockout_00") and type(effective) is int and effective <= tick:
            gcs_lockout = {"action_id": aid, "event_tick": action["tick"],
                           "effective_tick": effective}
        if aid.endswith("command_lockout_01") and type(effective) is int and effective <= tick:
            uav_lockout = {"action_id": aid, "event_tick": action["tick"],
                           "effective_tick": effective}
        if aid.endswith("command_lockout_00"):
            revoke_action_tick = action["tick"]
    if revoke_action_tick is None:
        raise ValueError("original command_lockout isolation action not found in executed actions")
    if gcs_lockout is None or uav_lockout is None:
        return {"schema_version": SCHEMA, "report": "owner_local_isolation_lockout",
                "reported": False, "reason": "owner_local_isolation_not_applied_yet",
                "revoked_epoch": revoked_movement_epoch(), "isolation_event_tick": revoke_action_tick,
                "gcs_lockout_effective_tick": (gcs_lockout or {}).get("effective_tick"),
                "uav_lockout_effective_tick": (uav_lockout or {}).get("effective_tick"),
                "evaluated_tick": tick}
    return {"schema_version": SCHEMA, "report": "owner_local_isolation_lockout",
            "reported": True, "reason": None, "revoked_epoch": revoked_movement_epoch(),
            "isolation_event_tick": gcs_lockout["event_tick"],
            "gcs_lockout_effective_tick": gcs_lockout["effective_tick"],
            "uav_lockout_effective_tick": uav_lockout["effective_tick"],
            "evaluated_tick": tick}


def find_motion_execution(engine, action_id, observation_tick):
    """The actual executed movement action and its real window (or None).

    Reads ``engine.executed_actions`` verbatim: a move_entity action executed at
    tick T creates its first real displacement in the interpolation segment
    [T*STEP_NS, (T+1)*STEP_NS), which is why the actual motion window starts at
    the executed tick and not before.
    """
    for action in engine.executed_actions:
        if action.get("action_id") == action_id and action.get("type") == "move_entity":
            result = action.get("result") or {}
            if result.get("status") != "ok":
                return None
            executed_tick = action["tick"]
            if observation_tick < executed_tick:
                return None
            return {"action_id": action_id, "entity_id": action.get("entity_id"),
                    "executed_tick": executed_tick,
                    "executed_ns": executed_tick * STEP_NS,
                    "path_length_m": result.get("path_length_m"),
                    "motion_window_start_ns": executed_tick * STEP_NS,
                    "motion_window_end_ns": (executed_tick + 1) * STEP_NS}
    return None


def bind_motion_to_admission(admission, motion_execution):
    """Causal gate from the admitted receipt to the actual physical motion.

    Fails closed: no admitted admission, no actual executed movement action at
    or after the admission barrier, or trajectory rows without real
    displacement all produce ``bound: False``. This gate exists because
    ``physical_motion_evidence: null`` in the admission record is "not
    evaluated here", never proof of execution; the two records together are the
    evidence pair. Trajectory rows are the actual frozen engine rows
    (entity_id/pos_enu/vel_mps per tick), filtered by the caller.
    """
    evidence = {"schema_version": SCHEMA, "gate": "motion_bound_to_admission",
                "message_id": admission.get("message_id"),
                "admission_decision": admission.get("decision"),
                "admission_ns": admission.get("admission_ns"),
                "motion_action_id": (motion_execution or {}).get("action_id"),
                "motion_executed_tick": (motion_execution or {}).get("executed_tick"),
                "path_length_m": (motion_execution or {}).get("path_length_m"),
                "displacement_m": None, "mean_speed_mps": None,
                "bound": False, "reason": None}
    if admission.get("decision") != "admitted":
        evidence["reason"] = "command_not_admitted"
        return evidence
    if motion_execution is None:
        evidence["reason"] = "movement_action_not_executed"
        return evidence
    if motion_execution["executed_ns"] < admission["admission_ns"]:
        evidence["reason"] = "motion_started_before_admission_barrier"
        return evidence
    evidence["bound"] = True
    return evidence


def summarize_motion_rows(rows, window_start_ns, window_end_ns):
    """Actual displacement/speed from real trajectory rows around a window.

    ``rows`` are the engine's frozen per-tick UAV rows (pos_enu, vel_mps, tick).
    Rows are instantaneous samples at their tick time; the displacement of the
    motion window [start, end) is the real displacement between the sample at
    ``start`` and the next sample at ``end`` (inclusive closing sample), and the
    mean speed is the displacement-weighted mean over the real intervals.
    Nothing is interpolated or fabricated beyond the engine's own linear
    interpolation between frozen rows.
    """
    picked = []
    for row in rows:
        if row["entity_id"] != UAV:
            continue
        row_ns = row["tick"] * STEP_NS
        if window_start_ns <= row_ns <= window_end_ns:
            picked.append(row)
    picked.sort(key=lambda r: r["tick"])
    if len(picked) < 2:
        return {"rows_used": len(picked), "displacement_m": None,
                "mean_speed_mps": None, "window_start_ns": window_start_ns,
                "window_end_ns": window_end_ns}
    displacement = 0.0
    interval_speeds = []
    previous = None
    for row in picked:
        x, y, z = row["pos_enu"]
        if previous is not None:
            segment = ((x - previous[0]) ** 2 + (y - previous[1]) ** 2
                       + (z - previous[2]) ** 2) ** 0.5
            dt_s = (row["tick"] - previous[3]) / 10.0
            displacement += segment
            interval_speeds.append(segment / dt_s)
        previous = (x, y, z, row["tick"])
    return {"rows_used": len(picked), "displacement_m": displacement,
            "mean_speed_mps": (sum(interval_speeds) / len(interval_speeds)),
            "interval_speeds_mps": interval_speeds,
            "window_start_ns": window_start_ns, "window_end_ns": window_end_ns}


def admission_barrier_tick(executed_tick):
    """Earliest original 5-tick action-grid boundary at or after a tick.

    The receiver controller evaluates on the authored 5-tick event grid; a
    candidate is a grid boundary exactly when ``tick % 5 == 0``. Example from
    the frozen trace: RX at 27.0023263 s falls inside segment (270, 271], so
    the first grid boundary after the RX instant is tick 275 (27.5 s).
    """
    candidate = executed_tick
    while candidate % 5 != 0:
        candidate += 1
    return candidate


def next_grid_tick_after_ns(rx_ns):
    """Earliest original 5-tick grid tick whose instant is strictly after an RX.

    Uses the actual 100 ms step: the RX instant is located inside one 0.1 s
    segment; the first grid boundary (tick multiple of 5) whose instant is
    >= the segment end. Examples from the actual coupled trace:
    RX 27.0023263e9 -> 275, RX 38.000214883e9 -> 385, RX 46.000223477e9 -> 465.
    """
    if type(rx_ns) is not int or rx_ns < 0:
        raise ValueError("rx_ns must be a nonnegative integer nanosecond instant")
    segment_end_tick = -(-rx_ns // STEP_NS)  # ceil: first sample at/after the RX
    return admission_barrier_tick(segment_end_tick)

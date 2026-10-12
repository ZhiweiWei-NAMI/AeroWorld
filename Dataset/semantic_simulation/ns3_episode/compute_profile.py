"""Compile declared legacy capacities into explicit local task assumptions.

Legacy instantaneous demand and queue-age slack are not task work, job memory,
arrival rates or generation deadlines. Never infer those fields from labels.
"""
from __future__ import annotations
from decimal import Decimal

from .compute_resource import VERSION


def _positive_integer(value, label):
    if type(value) not in (int, float, Decimal):
        raise ValueError(f'{label} must be numeric, not a string or boolean')
    number = Decimal(str(value))
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError(f'{label} must be an exact positive integer')
    return int(number)


def resolve_legacy_local_profile(*, episode_id, owner_id, node_category, horizon_ns,
                                 episode_seed, legacy_profile, defaults, source_ref):
    """Reuse exact category capacities/class; missing task mechanics need defaults.

    Defaults are required explicit input, not fallbacks hidden in the compiler.
    The existing schema has no per-owner override, CPU effective frequency,
    per-job cycles/memory, true arrivals, waiting-job limit or task deadline.
    A different source schema must be mapped explicitly, not guessed by alias.
    """
    if legacy_profile['model']['model_version'] != '1.6.0':
        raise ValueError('This explicit source adapter requires legacy model1.6.0')
    node = legacy_profile['compute']['node_profiles'][node_category]
    task = legacy_profile['compute']['task_profiles'][node_category]
    cores = _positive_integer(node['cpu_cores'], 'node.cpu_cores')
    if type(node['memory_mb']) not in (int, float) or not Decimal(str(node['memory_mb'])).is_finite():
        raise ValueError('node.memory_mb must be a finite numeric MB quantity')
    memory = _positive_integer(Decimal(str(node['memory_mb']))*1_000_000, 'node.memory_mb converted MB->byte')
    required = {'cpu_hz','cycles_per_job','job_memory_bytes','arrival_period_ns',
                'task_deadline_duration_ns','waiting_queue_limit_jobs','source_ref'}
    if set(defaults) != required:
        raise ValueError(f'Authored defaults require exactly {sorted(required)}')
    for key in required - {'source_ref','waiting_queue_limit_jobs'}:
        if type(defaults[key]) is not int or defaults[key] <= 0:
            raise ValueError(f'{key} must be an explicit positive integer')
    if type(defaults['waiting_queue_limit_jobs']) is not int or defaults['waiting_queue_limit_jobs'] < 0:
        raise ValueError('waiting_queue_limit_jobs must be nonnegative integer')
    if not isinstance(defaults['source_ref'], str) or not defaults['source_ref'].strip():
        raise ValueError('authored defaults require a nonempty source_ref')
    if not isinstance(task['task_kind'], str) or not task['task_kind'].strip():
        raise ValueError('existing task profile requires an explicit job class')
    origins = {
        'cpu_cores':{'status':'existing_authored_model_profile','source':f'{source_ref}#/compute/node_profiles/{node_category}/cpu_cores','unit':'core'},
        'memory_capacity_bytes':{'status':'derived_existing_authored_model_profile','source':f'{source_ref}#/compute/node_profiles/{node_category}/memory_mb','unit':'byte','mapping':'source MB*1000000, not MiB'},
        'job_class':{'status':'existing_authored_task_profile','source':f'{source_ref}#/compute/task_profiles/{node_category}/task_kind','unit':'1'},
        'seed':{'status':'existing_episode_manifest','source':f'aw_data/render_ready_episodes/{episode_id}/episode_manifest.json#/seed','unit':'1'}}
    origins['burst_jobs']={'status':'authored_fixed_workload_policy',
        'source':'Dataset/semantic_simulation/ns3_episode/compute_profile.py#resolve_legacy_local_profile',
        'unit':'job/batch','assumption':'This source adapter explicitly generates one job per periodic arrival; source has no job-count arrival mechanism.'}
    for key in required-{'source_ref'}:
        origins[key]={'status':'authored_default_missing_task_mechanism','source':defaults['source_ref'],
                      'unit':'ns' if key.endswith('_ns') else 'Hz' if key=='cpu_hz' else 'byte' if key.endswith('_bytes') else 'cycle' if key=='cycles_per_job' else 'job'}
    return {'schema_version':'p09.compute.resource-config/v2','producer_version':VERSION,
        'episode_id':episode_id,'duration_ns':horizon_ns,'owner_entity_ids':[owner_id],
        'seed':episode_seed,'seed_use':'not_used_deterministic_workload','profile':'authored_local_v1',
        'source_ref':f'aw_data/render_ready_episodes/{episode_id}/truth_frames.jsonl',
        'render_inputs_changed':False,'field_provenance':origins,
        'local_unavailability_intervals_ns':[],
        'local_unavailability_policy':'cancel_local_jobs_reject_arrivals_then_resume',
        'compute':{'cpu_cores':cores,'cpu_hz':defaults['cpu_hz'],'cycles_per_job':defaults['cycles_per_job'],
            'memory_capacity_bytes':memory,'job_memory_cycle_bytes':[defaults['job_memory_bytes']],
            'job_class':task['task_kind'],'arrival_period_ns':defaults['arrival_period_ns'],'burst_jobs':1,
            'waiting_queue_limit_jobs':defaults['waiting_queue_limit_jobs'],
            'task_deadline_duration_ns':defaults['task_deadline_duration_ns'],
            'arrival_process':'declared_periodic_bursts','scheduling':'strict_fifo_nonpreemptive',
            'service_randomness':'none','memory_scope':'executing_jobs_only; queued input buffers not modeled'},
        'legacy_quantities_not_mapped':{'cpu_cores_demand':task['cpu_cores'],'gpu_units_demand':task['gpu_units'],
            'memory_mb_aggregate_demand':task['memory_mb'],'queue_segment_slack_ticks':task['deadline_slack_ticks'],
            'aggregate_queue_depth_limit':node['max_queue_depth'],
            'reason':'aggregate instantaneous demand/queue age is not perjob work/memory/arrival/deadline/waiting count'},
        'unavailable_mechanisms':{'gpu':'UNKNOWN no workload/service profile',
            'offload':'UNKNOWN no request-service-response producer',
            'migration':'UNKNOWN no checkpoint-transfer-resume producer'},
        'assumption_scope':'Controlled authored local CPU workload; no hardware calibration or physical motion effect'}

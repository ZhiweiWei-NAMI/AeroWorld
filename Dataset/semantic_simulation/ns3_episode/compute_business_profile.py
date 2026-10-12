"""Explicit task-class requests sharing the existing owner resource scheduler.

No target event, predicate or future motion value determines arrivals or work.
Missing sensor cadence must be authored; render/capture cadence is not a sensor
request binding. Configured CPU frequency/work remain uncalibrated assumptions.
"""
from decimal import Decimal

from .compute_resource import VERSION


def compile_business_profile(episode, scene, workload, regime, owner, *, scene_ref, workload_ref, legacy, legacy_ref):
    if workload['schema_version'] != 'p09.compute.business-request-profile/v1':
        raise ValueError('explicit business request profile v1 required')
    if workload['adoption'] != 'MECHANISM_CANDIDATE_NOT_FINAL_RELEASE':
        raise ValueError('this compiler only prepares declared mechanism candidates')
    if legacy['model']['model_version'] != '1.6.0':
        raise ValueError('capacity mapping is explicitly defined for source1.6.0')
    if scene['scenario_id']+'__seed'+str(episode['seed']).zfill(2) != episode['episode_id']:
        raise ValueError('scene and source episode/seed differ')
    entities = [e for e in scene['entities'] if e['entity_id'] == owner]
    nodes = [n for n in episode['nodes'] if n['owner'] == owner]
    if len(entities) != 1 or not nodes or {n['node_type'] for n in nodes} != {entities[0]['category']}:
        raise ValueError('exact scene owner and normalized source epochs required')
    entity = entities[0]
    category = entity['category']
    capacity = legacy['compute']['node_profiles'][category]
    source_class = legacy['compute']['task_profiles'][category]['task_kind']
    if type(capacity['memory_mb']) not in (int, float):
        raise ValueError('source memory_mb must be numeric MB, not string/bool')
    task_id = entity['initial_state']['task_id']
    if not isinstance(task_id, str) or not task_id:
        raise ValueError('scene task identity required; no owner aliasing')
    if type(capacity['cpu_cores']) not in (int, float) or not Decimal(str(capacity['cpu_cores'])).is_finite() \
            or Decimal(str(capacity['cpu_cores'])) <= 0 or int(capacity['cpu_cores']) != capacity['cpu_cores']:
        raise ValueError('source CPU cores must be exact positive integer')
    memory = Decimal(str(capacity['memory_mb']))*1_000_000
    if not memory.is_finite() or memory <= 0 or memory != memory.to_integral_value():
        raise ValueError('source memory MB must map to exact positive bytes')
    common = workload['executor_assumptions']
    if type(common['cpu_hz']) is not int or common['cpu_hz'] <= 0:
        raise ValueError('cpu_hz must be explicit positive integer Hz')
    if type(common['waiting_queue_limit_jobs']) is not int or common['waiting_queue_limit_jobs'] < 0:
        raise ValueError('waiting_queue_limit_jobs must be explicit nonnegative integer')
    classes = workload['regimes'][regime]['classes']
    required = {'class_id','parent_task_kind','arrival_period_ns','cycles','memory_bytes',
                'deadline_duration_ns','arrival_provenance','work_provenance'}
    if not classes or len({c['class_id'] for c in classes}) != len(classes):
        raise ValueError('distinct nonempty task classes required')
    jobs = []
    for cls in classes:
        if set(cls) != required or cls['parent_task_kind'] != source_class:
            raise ValueError('exact task class/source parent contract required')
        for field in ('arrival_period_ns','cycles','memory_bytes','deadline_duration_ns'):
            if type(cls[field]) is not int or cls[field] <= 0:
                raise ValueError(f'{field} must be explicit positive integer')
        if cls['cycles']*1_000_000_000 % common['cpu_hz']:
            raise ValueError('work/frequency must resolve to integer service ns')
        for key in ('arrival_provenance','work_provenance'):
            if cls[key]['status'] != 'authored_missing_source_assumption' or not cls[key]['reason']:
                raise ValueError('current scene lacks these mechanics; do not claim observations')
        for node in nodes:
            end = episode['duration_ns'] if node['death_ns'] is None else min(node['death_ns'],episode['duration_ns'])
            for number, at in enumerate(range(node['birth_ns'],end,cls['arrival_period_ns'])):
                jobs.append({'job_id':f"{owner}:{node['life_epoch']}:{cls['class_id']}:{number}",
                    'owner_entity_id':owner,'actor_epoch':node['life_epoch'],'arrival_ns':at,
                    'cycles':cls['cycles'],'memory_bytes':cls['memory_bytes'],
                    'deadline_duration_ns':cls['deadline_duration_ns'],'job_class':cls['class_id'],
                    'workload_source_ref':workload_ref+'#/regimes/'+regime+'/classes'})
    jobs.sort(key=lambda j:(j['arrival_ns'],j['job_class'],j['job_id']))
    offered = sum(1_000_000_000/c['arrival_period_ns'] for c in classes)
    busy = sum(c['cycles']/common['cpu_hz']*1_000_000_000/c['arrival_period_ns'] for c in classes)
    return {'schema_version':'p09.compute.resource-config/v3','producer_version':VERSION,
        'episode_id':episode['episode_id'],'duration_ns':episode['duration_ns'],'owner_entity_ids':[owner],
        'seed':episode['seed'],'seed_use':'not_used_deterministic_workload','profile':regime,
        'adoption':'MECHANISM_CANDIDATE_NOT_FINAL_RELEASE','render_inputs_changed':False,
        'source_ref':episode['source_ref'],'scene_binding':{'scene_ref':scene_ref,'entity_id':owner,
            'scene_task_id':task_id,'parent_task_kind':source_class,'logical_asset_id':entity['logical_asset_id']},
        'field_provenance':{'cpu_cores':{'status':'existing_authored_model_profile','source':legacy_ref+'#/compute/node_profiles/'+category+'/cpu_cores','unit':'core'},
            'memory_capacity_bytes':{'status':'derived_existing_authored_model_profile','source':legacy_ref+'#/compute/node_profiles/'+category+'/memory_mb','unit':'byte','mapping':'MB*1000000'},
            'cpu_hz':{'status':'authored_missing_source_assumption','source':workload_ref+'#/executor_assumptions/cpu_hz','unit':'Hz'},
            'task_manifest':{'status':'derived_authored_request_profile','source':workload_ref,'unit':'ns/cycle/byte','arrival_clock':'absolute episode simulation ns; reanchor each observed life epoch'}},
        'local_unavailability_intervals_ns':[i for i in episode['local_unavailability_intervals_ns'] if i['owner_entity_id']==owner],
        'local_unavailability_policy':'cancel_local_jobs_reject_arrivals_then_resume',
        'compute':{'cpu_cores':int(capacity['cpu_cores']),'cpu_hz':common['cpu_hz'],'memory_capacity_bytes':int(memory),
            'waiting_queue_limit_jobs':common['waiting_queue_limit_jobs'],'arrival_process':'declared_task_manifest',
            'task_manifest':jobs,'scheduling':'strict_fifo_nonpreemptive','service_randomness':'none',
            'memory_scope':'executing_jobs_only; queued input buffers not modeled'},
        'theory':{'offered_jobs_per_s':offered,'uninterrupted_mean_busy_cores':busy,
            'offered_cpu_utilization':busy/capacity['cpu_cores'],
            'CPU_only_mixture_capacity_jobs_per_s':capacity['cpu_cores']/(busy/offered),
            'max_executing_memory_bound_bytes':int(capacity['cpu_cores'])*max(c['memory_bytes'] for c in classes),
            'scope':'steady-state lower resource constraint; outage and FIFO burst effects measured separately'},
        'unavailable_mechanisms':{'gpu':'UNKNOWN no workload/service profile','offload':'UNKNOWN no request-response transport','migration':'UNKNOWN no checkpoint-resume producer'},
        'classes':classes,'limitations':['Not sensor measurements or calibrated hardware','No compute-to-flight policy; motion unchanged',
            'Normal/congested are predeclared request-rate sensitivity controls, not final210workload distribution']}

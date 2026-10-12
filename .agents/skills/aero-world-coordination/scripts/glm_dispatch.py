"""Dispatch explicit parallel leaves; continue length-limited turns in place.

Tasks JSON is an array of {name, prompt, source_files, writable_files}. Paths are
project-relative. --brief supplies shared text. Each leaf receives the same brief
and its named source contents. --max-tokens, when explicit, applies to every leaf
in this process; the default advertised output limit is 131072. This is a resolver
limit, not verified live acceptance. Reasoning effort stays at provider default.
Results contain public
text only, route metadata, usage and tool counts. 
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, field
import json
import io
import math
from pathlib import Path
import re
import sys
import time
import uuid

MODEL = 'glm-5.3-flash'
PROVIDER = 'workbuddy'
MAX_TOKENS = 131072
LAUNCHER = 'dsh'
ROOT = Path('/mnt/data2/weizhiwei/AERO_WORLD')


class DispatchError(Exception):
    """A surfaced protocol, configuration or runtime failure."""


class Frames:
    """Incremental NDJSON framing; chunk size is independent of message length."""
    def __init__(self):
        self.buffer = bytearray()
        self.scan = 0

    def feed(self, chunk):
        self.buffer.extend(chunk)
        records = []
        start = 0
        while True:
            end = self.buffer.find(b'\n', self.scan)
            if end < 0:
                self.scan = len(self.buffer)
                break
            raw = self.buffer[start:end]
            try:
                item = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DispatchError('Invalid stdout JSON frame; raw content withheld') from exc
            if not isinstance(item, dict) or item.get('jsonrpc') != '2.0':
                raise DispatchError('Invalid JSON-RPC envelope')
            records.append(item)
            start = end + 1
            self.scan = start
        if start:
            del self.buffer[:start]
            self.scan -= start
        return records

    def finish(self):
        if self.buffer:
            raise DispatchError('EOF inside stdout JSON frame')


class Diagnostics:
    """Drain stderr while retaining only bounded, redacted diagnostic excerpts."""
    def __init__(self, limit=4096):
        self.limit = limit
        self.lines = []
        self.bytes = 0
        self.omitted = 0
        self.pending = bytearray()
        self.discard = False

    @staticmethod
    def redact(text):
        if re.search(r'proxy|credential|authorization|password|secret|api.?key|access.?token|endpoint|base.?url', text, re.I):
            return '[sensitive diagnostic line redacted]'
        text = re.sub(r'(?:https?|socks\d?)://\S+', '[URL redacted]', text, flags=re.I)
        text = re.sub(r'\bBearer\s+\S+|\bsk-[\w-]+', '[token redacted]', text, flags=re.I)
        text = re.sub(r'\b(?:[\w.-]+|\[[0-9a-f:]+\]):\d{2,5}\b', '[address redacted]', text, flags=re.I)
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[address redacted]', text)
        return text

    def append(self, text):
        text = self.redact(text)
        remaining = self.limit - sum(len(line) + 1 for line in self.lines)
        if len(text) + 1 <= remaining:
            self.lines.append(text)
        else:
            self.omitted += 1

    def feed(self, chunk):
        self.bytes += len(chunk)
        for part in chunk.splitlines(keepends=True):
            ended = part.endswith(b'\n')
            if not self.discard:
                self.pending.extend(part)
                if len(self.pending) > self.limit:
                    self.pending.clear()
                    self.discard = True
            if ended:
                if self.discard:
                    self.append('[oversized stderr line omitted]')
                else:
                    self.append(self.pending.decode('utf-8', errors='replace').rstrip('\r\n'))
                self.pending.clear()
                self.discard = False

    def finish(self):
        if self.discard:
            self.append('[oversized stderr line omitted]')
        elif self.pending:
            self.append(self.pending.decode('utf-8', errors='replace'))
        self.pending.clear()

    def result(self):
        return {'bytes': self.bytes, 'excerpt': self.lines, 'omitted_lines': self.omitted}


def numeric_usage(value):
    """Only numeric usage metadata; never retain arbitrary provider strings."""
    if isinstance(value, dict):
        return {key: parsed for key, item in value.items()
                if (parsed := numeric_usage(item)) is not None}
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def terminal_reason(reason):
    """Retain bounded error metadata without provider payloads or addresses."""
    if not isinstance(reason, dict) or not isinstance(reason.get('kind'), str):
        raise DispatchError('Terminal reason must be an object with kind')
    end = {'kind': reason['kind']}
    if 'error' in reason:
        error = reason['error']
        if not isinstance(error, dict):
            raise DispatchError('Terminal error must be an object')
        end['error'] = {}
        for key in ('code', 'message', 'requestId'):
            if key in error:
                if not isinstance(error[key], (str, int)) or isinstance(error[key], bool):
                    raise DispatchError(f'Terminal error {key} must be text or integer')
                end['error'][key] = Diagnostics.redact(str(error[key]))[:512]
        if 'status' in error:
            if not isinstance(error['status'], int) or isinstance(error['status'], bool):
                raise DispatchError('Terminal error status must be an integer')
            end['error']['status'] = error['status']
    return end


@dataclass
class TaskResult:
    name: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resume_session_id: str | None = None
    public_messages: list = field(default_factory=list)
    headers: list = field(default_factory=list)
    usage: list = field(default_factory=list)
    calls: Counter = field(default_factory=Counter)
    call_names: dict = field(default_factory=dict)
    results: Counter = field(default_factory=Counter)
    tool_errors: Counter = field(default_factory=Counter)
    terminal: dict | None = None
    turn_ends: list = field(default_factory=list)
    last_length_progress: tuple | None = None
    requested_max_tokens: int = MAX_TOKENS
    failures: list = field(default_factory=list)
    active_turn: int | None = None
    last_turn: int | None = None
    admitted_messages: set = field(default_factory=set)
    consumed_messages: set = field(default_factory=set)
    queued_messages: set = field(default_factory=set)
    dispatched_calls: set = field(default_factory=set)
    completed_calls: set = field(default_factory=set)
    tool_identity_events: list = field(default_factory=list)
    source_events: dict = field(default_factory=dict)
    verified_replacements: list = field(default_factory=list)
    replacement_count: int = 0
    resume_event_source: str | None = None

    def declare_call(self, call_id, name):
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise DispatchError('Tool declaration requires nonempty string identity and name')
        if call_id in self.call_names and self.call_names[call_id] != name:
            raise DispatchError('Tool declaration name differs for the same identity')
        self.call_names[call_id] = name

    def verify_replacement(self, event):
        """A pruner replacement cites its original result, not a new call."""
        op = event['surfaceOp']
        sources = event['sourceEventSeqs']
        if (op.get('op') != 'replace' or not isinstance(sources, list) or len(sources) != 1
                or not isinstance(sources[0], int) or isinstance(sources[0], bool)
                or op.get('startSeq') != sources[0] or op.get('endSeq') != sources[0]
                or not isinstance(event.get('seq'), int) or sources[0] >= event['seq']):
            raise DispatchError('Tool replacement has invalid source range')
        source = self.source_events.get(sources[0])
        if source is None or source['type'] != 'tool/result':
            raise DispatchError('Tool replacement has no genuine source result metadata')
        data = event['data']
        blocks = [block for block in data['message']['content'] if block['type'] == 'tool-result']
        identities = [(block['toolCallId'], block['isError']) for block in blocks]
        expected = [(call['callId'], call['isError']) for call in source['calls']]
        if (identities != expected or data['turn'] != source['turn']
                or data['step'] != source['step']):
            raise DispatchError('Tool replacement identity or turn differs from its source result')
        return source

    def index_tool_event(self, event, declarations):
        """Keep source correlation metadata without retaining tool contents."""
        if event['type'] not in ('tool/call', 'tool/result') or 'seq' not in event:
            return
        seq, data = event['seq'], event['data']
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0 or seq in self.source_events:
            raise DispatchError('Invalid or duplicate source event sequence')
        if event['type'] == 'tool/call':
            call_id, name = data['callId'], data['name']
            if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
                raise DispatchError('Historical tool declaration lacks identity or name')
            if call_id in declarations and declarations[call_id] != name:
                raise DispatchError('Historical tool declaration has conflicting names')
            declarations[call_id] = name
            calls = [{'callId': call_id, 'name': name}]
        elif isinstance(event.get('surfaceOp'), dict):
            self.source_events[seq] = {**self.verify_replacement(event),
                                      'source_seqs': list(event['sourceEventSeqs'])}
            return
        else:
            sources = event['sourceEventSeqs']
            if not isinstance(sources, list) or not sources:
                raise DispatchError('Source tool result lacks its declaration sequence')
            source_calls = {}
            for source_seq in sources:
                source = self.source_events.get(source_seq)
                if source is None or source['type'] != 'tool/call' or source_seq >= seq:
                    raise DispatchError('Source tool result references an undeclared call event')
                source_calls.update((call['callId'], call['name']) for call in source['calls'])
            calls = []
            for block in data['message']['content']:
                if block['type'] != 'tool-result':
                    continue
                call_id = block['toolCallId']
                if call_id not in source_calls or declarations.get(call_id) != source_calls[call_id]:
                    raise DispatchError('Source tool result identity differs from its declaration')
                calls.append({'callId': call_id, 'name': source_calls[call_id], 'isError': block['isError']})
        self.source_events[seq] = {'type': event['type'], 'turn': data['turn'],
                                   'step': data['step'], 'calls': calls,
                                   'source_seqs': list(event.get('sourceEventSeqs', []))}

    def record_tool_identity(self, event, envelope):
        """Record only envelope keys and correlation metadata, never payloads."""
        kind, data = event['type'], event['data']
        if kind not in ('assistant/message', 'tool/call', 'tool/result'):
            return
        message = data.get('message', {})
        blocks = message.get('content', [])
        relevant = [block for block in blocks if block.get('type') in ('tool-call', 'tool-result')]
        if kind == 'assistant/message' and not relevant:
            return

        def keys(value):
            return [Diagnostics.redact(str(key))[:64] for key in value][:32]

        def identity(value):
            return {key: Diagnostics.redact(value[key])[:128]
                    for key in ('id', 'callId', 'toolCallId', 'tool_call_id', 'name')
                    if isinstance(value.get(key), str)}

        record = {'session_id': self.session_id, 'type': kind, 'event_keys': keys(event),
                  'data_keys': keys(data), 'identity': identity(data),
                  'message_keys': keys(message),
                  'block_count': len(relevant),
                  'blocks': [{'type': block['type'], 'keys': keys(block), 'identity': identity(block)}
                             for block in relevant[:32]]}
        for key in ('seq', 'time'):
            if isinstance(event.get(key), int):
                record[key] = event[key]
        if 'sourceEventSeqs' in event:
            sources = event['sourceEventSeqs']
            if isinstance(sources, list) and all(isinstance(seq, int) for seq in sources):
                record['sourceEventSeqs'] = sources[:32]
        if isinstance(event.get('surfaceOp'), dict):
            record['surfaceOp'] = {key: event['surfaceOp'][key]
                                   for key in ('op', 'startSeq', 'endSeq') if key in event['surfaceOp']}
        elif isinstance(event.get('surfaceOp'), str):
            record['surfaceOp'] = event['surfaceOp']
        for key in ('turn', 'step'):
            if isinstance(data.get(key), int):
                record[key] = data[key]
        if envelope is not None:
            record['envelope_keys'] = keys(envelope)
            record['params_keys'] = keys(envelope['params'])
        self.tool_identity_events.append(record)
        del self.tool_identity_events[:-32]

    def admit(self, message_id):
        if not isinstance(message_id, str) or not message_id:
            raise DispatchError('Prompt acknowledgment lacks a message identity')
        if message_id in self.admitted_messages:
            raise DispatchError('Duplicate prompt message identity')
        self.admitted_messages.add(message_id)
        if message_id not in self.consumed_messages:
            self.queued_messages.add(message_id)

    def settled(self):
        # An errored turn cannot consume its queued follow-ups. Preserve them
        # in the result instead of keeping the SDK bridge open forever.
        failed = bool(self.failures) or (self.terminal is not None
                                         and self.terminal.get('kind') != 'completed')
        return (self.active_turn is None and (failed or not self.queued_messages)
                and (self.terminal is not None or bool(self.failures)))

    def event(self, event, envelope=None):
        self.record_tool_identity(event, envelope)
        kind, data = event['type'], event['data']
        if kind == 'turn/start':
            turn = data['turn']
            if (not isinstance(turn, int) or isinstance(turn, bool) or turn < 1
                    or self.active_turn is not None
                    or (self.last_turn is not None and turn <= self.last_turn)):
                raise DispatchError('Invalid or overlapping turn start')
            self.active_turn = self.last_turn = turn
            self.terminal = None
        elif kind == 'user/message':
            message_id = data['id']
            if not isinstance(message_id, str) or not message_id:
                raise DispatchError('Consumed user message lacks an identity')
            self.consumed_messages.add(message_id)
            self.queued_messages.discard(message_id)
        elif kind == 'request/header':
            config = data['header']['config']
            route = {key: config[key] for key in ('provider', 'model', 'maxTokens', 'reasoningEffort') if key in config}
            self.headers.append(route)
            print(json.dumps({'event': 'request_header', 'task': self.name,
                              'session_id': self.session_id, 'route': route}), flush=True)
            if route.get('provider') != PROVIDER or route.get('model') != MODEL:
                raise DispatchError('Request header provider/model differs from explicit request')
        elif kind == 'assistant/message':
            for block in data['message']['content']:
                if block['type'] == 'tool-call':
                    self.declare_call(block['id'], block['name'])
            texts = [block['text'] for block in data['message']['content'] if block['type'] == 'text' and block['text'].strip()]
            if texts:
                self.public_messages.append('\n'.join(texts))
            if 'usage' in data:
                self.usage.append(numeric_usage(data['usage']))
        elif kind == 'tool/call':
            name, call_id = data['name'], data['callId']
            if call_id in self.dispatched_calls:
                raise DispatchError('Duplicate tool call identity')
            self.declare_call(call_id, name)
            self.dispatched_calls.add(call_id)
            self.calls[name] += 1
            self.index_tool_event(event, self.call_names)
        elif kind == 'tool/result':
            if isinstance(event.get('surfaceOp'), dict):
                # SDK history compaction edits an old transcript result. It
                # neither executes a new tool nor completes a live call.
                self.replacement_count += 1
                self.verified_replacements.append({
                    'seq': event['seq'], 'source_seqs': event['sourceEventSeqs'],
                    'classification': 'sdk-history-compaction',
                    'turn': data['turn'], 'step': data['step'],
                    'tool_call_ids': [block['toolCallId']
                        for block in data['message']['content']
                        if block['type'] == 'tool-result']})
                del self.verified_replacements[:-32]
                return
            self.index_tool_event(event, self.call_names)
            for block in data['message']['content']:
                if block['type'] == 'tool-result':
                    call_id = block['toolCallId']
                    if call_id not in self.call_names:
                        metadata = {key: data[key] for key in ('turn', 'step') if key in data}
                        metadata['callId'] = Diagnostics.redact(str(call_id))[:128]
                        raise DispatchError('Tool result has no declared call identity: '
                                            + json.dumps(metadata))
                    name = self.call_names[call_id]
                    if call_id in self.completed_calls:
                        raise DispatchError('Duplicate tool result identity')
                    self.completed_calls.add(call_id)
                    self.results[name] += 1
                    if block['isError']:
                        self.tool_errors[name] += 1
        elif kind == 'turn/end':
            if self.active_turn is None or data['turn'] != self.active_turn:
                raise DispatchError('Terminal event does not match an active turn')
            self.active_turn = None
            reason = data['reason']
            end = terminal_reason(reason)
            self.turn_ends.append({**end, 'tool_calls': sum(self.calls.values()),
                                   'tool_results': sum(self.results.values()),
                                   'public_messages': len(self.public_messages), 'usage_records': len(self.usage)})
            if reason['kind'] == 'max-tokens':
                progress = (sum(self.calls.values()) + sum(self.results.values()), frozenset(self.public_messages))
                if self.last_length_progress == progress:
                    self.terminal = {'kind': 'continuation-stalled'}
                    self.failures.append('Length continuation produced no new tool activity or public output')
                else:
                    self.last_length_progress = progress
                    return True
            else:
                self.terminal = end

    def compact(self):
        return {'name': self.name, 'session_id': self.session_id, 'resume_session_id': self.resume_session_id,
                'requested': {'provider': PROVIDER, 'model': MODEL, 'maxTokens': self.requested_max_tokens},
                'request_headers': self.headers, 'public_messages': self.public_messages,
                'terminal': self.terminal, 'turn_ends': self.turn_ends,
                'success': self.terminal == {'kind': 'completed'} and not self.failures and self.settled(),
                'tool_calls': dict(self.calls), 'tool_results': dict(self.results),
                'tool_call_names': dict(self.call_names),
                'dispatched_tool_call_ids': sorted(self.dispatched_calls),
                'completed_tool_call_ids': sorted(self.completed_calls),
                'pending_tool_call_ids': sorted(self.call_names.keys() - self.completed_calls),
                'unconsumed_prompt_ids': sorted(self.queued_messages),
                'tool_identity_events': self.tool_identity_events,
                'resume_event_source': self.resume_event_source,
                'history_compaction_count': self.replacement_count,
                'history_compactions': self.verified_replacements,
                'tool_errors': dict(self.tool_errors), 'usage': self.usage, 'failures': self.failures}


def load_tasks(path, root, brief):
    tasks = json.loads(path.read_text())
    if not isinstance(tasks, list) or not tasks:
        raise DispatchError('Tasks must be a nonempty JSON array')
    names, owners, prepared = set(), set(), []
    for task in tasks:
        if not isinstance(task, dict):
            raise DispatchError('Each task must be a JSON object')
        if set(task) != {'name', 'prompt', 'source_files', 'writable_files'}:
            raise DispatchError('Task keys must be name, prompt, source_files, writable_files')
        name = task['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise DispatchError('Task names must be unique safe filenames')
        names.add(name)
        if not isinstance(task['prompt'], str) or not task['prompt'].strip():
            raise DispatchError('Task prompt must be nonempty text')
        paths = {}
        for key in ('source_files', 'writable_files'):
            if not isinstance(task[key], list) or any(not isinstance(p, str) for p in task[key]):
                raise DispatchError('File declarations must be arrays of project-relative paths')
            paths[key] = []
            for item in task[key]:
                path = root / item
                if Path(item).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
                    raise DispatchError('File declarations must stay inside the project')
                paths[key].append(path)
        current = {p.resolve() for p in paths['writable_files']}
        if current & owners:
            raise DispatchError('Parallel leaves have overlapping file ownership')
        owners.update(current)
        sources = '\n'.join(f'\nSource {p.relative_to(root)}:\n{p.read_text()}' for p in paths['source_files'])
        prompt = ('Read applicable AGENTS. You are one leaf; do not delegate or invoke other models. '
                  'You are not alone in the codebase. Preserve others\' edits. Run only as weizhiwei; '
                  'no privilege escalation or credential/settings access. Write only declared files.\n'
                  f'Shared brief:\n{brief}\nAssignment:\n{task["prompt"]}\n'
                  f'Writable files: {json.dumps(task["writable_files"])}\nRead-only source context:{sources}')
        prepared.append((TaskResult(name), prompt))
    return prepared


def load_resume_events(path, states):
    """Read only correlation metadata from an explicitly supplied session log."""
    with path.open('rb') as handle:
        if path.suffix == '.zstd':
            import zstandard
            stream = zstandard.ZstdDecompressor().stream_reader(handle)
        else:
            stream = handle
        with io.TextIOWrapper(stream, encoding='utf-8') as records:
            header = json.loads(next(records))
            session_id = header['id']
            if session_id not in states or states[session_id].resume_session_id != session_id:
                raise DispatchError('Resume event source does not belong to an assigned resumed session')
            state = states[session_id]
            if state.resume_event_source is not None or state.source_events:
                raise DispatchError('Resume event source supplied more than once')
            declarations = {}
            for line in records:
                state.index_tool_event(json.loads(line), declarations)
            state.resume_event_source = str(path)


async def run(prepared, root, max_tokens=MAX_TOKENS, command=None, control_file=None):
    """One runtime, concurrent leaves, all failures surfaced; command is fixture-only."""
    for state, _ in prepared:
        state.requested_max_tokens = max_tokens
    try:
        process = await asyncio.create_subprocess_exec(
            *(command if command is not None else [LAUNCHER, '--profile', 'glm_sdk']),
            cwd=str(root), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
    except OSError as exc:
        failure = f'Runtime launch failed: {type(exc).__name__}'
        for state, _ in prepared:
            state.terminal = {'kind': 'runtime-launch-error'}
            state.failures.append(failure)
        return {'tasks': [state.compact() for state, _ in prepared], 'runtime_exit': None,
                'supervisor_errors': [failure], 'stderr': Diagnostics().result()}, 1
    diagnostics = Diagnostics()
    print(json.dumps({'event': 'runtime_started', 'pid': process.pid,
                      'tasks': [{'name': state.name, 'session_id': state.session_id}
                                for state, _ in prepared]}), flush=True)

    async def drain_stderr():
        while chunk := await process.stderr.read(65536):
            diagnostics.feed(chunk)
        diagnostics.finish()

    async def drain_remaining_stdout():
        """Release pipe backpressure after protocol parsing has stopped."""
        while await process.stdout.read(65536):
            pass

    stderr_task = asyncio.create_task(drain_stderr())
    states = {state.session_id: state for state, _ in prepared}
    pending = {1: 'initialize'}
    next_request_id = len(prepared) + 2
    initialized = False
    shutdown_sent = False
    shutdown_ack = False
    supervisor_errors = []
    steering_requests = set()
    activity_times = {}

    async def send(request_id, method, params):
        process.stdin.write((json.dumps({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params}) + '\n').encode())
        await process.stdin.drain()

    async def steer_from_file():
        """Queue follow-ups on existing agents, registering each RPC response."""
        nonlocal next_request_id
        offset = 0
        by_name = {state.name: state for state, _ in prepared}
        while process.returncode is None:
            if initialized and not shutdown_sent and control_file.exists():
                with control_file.open() as handle:
                    handle.seek(offset)
                    while line := handle.readline():
                        if not line.endswith('\n'):
                            break
                        offset = handle.tell()
                        instruction = json.loads(line)
                        state = by_name[instruction['task']]
                        if state.terminal is not None or state.failures:
                            raise DispatchError('Follow-up target is no longer active')
                        text = instruction['text']
                        if not isinstance(text, str) or not text.strip():
                            raise DispatchError('Follow-up requires nonempty public text')
                        request_id = next_request_id
                        next_request_id += 1
                        pending[request_id] = state.session_id
                        steering_requests.add(request_id)
                        await send(request_id, 'session/prompt', {
                            'sessionId': state.session_id,
                            'contentBlocks': [{'type': 'text', 'text': text}]})
            await asyncio.sleep(.5)

    async def consume():
        nonlocal initialized, shutdown_sent, shutdown_ack, next_request_id
        parser = Frames()
        async def shutdown_if_done():
            nonlocal shutdown_sent
            if initialized and not shutdown_sent and not pending and all(s.settled() for s in states.values()):
                pending[0] = 'shutdown'
                await send(0, 'shutdown', None)
                shutdown_sent = True
        while chunk := await process.stdout.read(65536):
            for record in parser.feed(chunk):
                if 'id' in record:
                    request_id = record['id']
                    if request_id not in pending:
                        raise DispatchError('RPC response has unknown request identity')
                    target = pending.pop(request_id)
                    if 'error' in record:
                        error = record['error']
                        description = f'RPC error code={error.get("code")}: ' + Diagnostics.redact(str(error.get('message', '')))[:512]
                        if target in states:
                            states[target].failures.append(description)
                        else:
                            raise DispatchError(description)
                    elif 'result' not in record:
                        raise DispatchError('RPC response lacks result/error')
                    elif target == 'initialize':
                        initialized = True
                        print(json.dumps({'event': 'initialized', 'pid': process.pid}), flush=True)
                        for index, (state, prompt) in enumerate(prepared, 2):
                            pending[index] = state.session_id
                            prompt_params = {'sessionId': state.session_id, 'contentBlocks': [{'type': 'text', 'text': prompt}]}
                            if state.resume_session_id is not None:
                                prompt_params['resumeSessionId'] = state.resume_session_id
                            await send(index, 'session/prompt', prompt_params)
                    elif target == 'shutdown':
                        shutdown_ack = True
                    elif target in states:
                        states[target].admit(record['result']['messageId'])
                    if request_id in steering_requests:
                        steering_requests.remove(request_id)
                        print(json.dumps({'event': 'followup_admitted', 'session_id': target,
                                          'error': 'error' in record}), flush=True)
                elif record.get('method') == 'session.event':
                    params = record['params']
                    if params['sessionId'] not in states:
                        raise DispatchError('Event belongs to an undeclared session')
                    state = states[params['sessionId']]
                    kind = params['event']['type']
                    if state.event(params['event'], record):
                        request_id = next_request_id
                        next_request_id += 1
                        pending[request_id] = state.session_id
                        await send(request_id, 'session/prompt', {
                            'sessionId': state.session_id, 'contentBlocks': [{'type': 'text', 'text':
                            'Continue the interrupted task in this same session using existing agent/tool state. '
                            'Do not restart or repeat completed work. Finish the code deliverable and checks, '
                            'or report a concrete blocker explicitly. Preserve the original scope and ownership.'}]})
                    if kind == 'request/header':
                        print(json.dumps({'event': 'actual_route', 'task': state.name,
                                          'route': state.headers[-1]}), flush=True)
                    if kind in ('tool/call', 'tool/result', 'turn/end') and (
                            kind == 'turn/end' or time.monotonic() - activity_times.get(state.name, 0) > 30):
                        activity_times[state.name] = time.monotonic()
                        print(json.dumps({'event': 'task_activity', 'task': state.name,
                                          'time_ms': int(time.time() * 1000),
                                          'tool_calls': dict(state.calls),
                                          'tool_results': dict(state.results),
                                          'terminal': state.terminal}), flush=True)
                elif record.get('method') == 'session.status':
                    pass  # Lifecycle notification is not a terminal success authority.
                elif record.get('method') in ('subagent.started', 'subagent.finished'):
                    raise DispatchError('Leaf delegated despite the explicit no-delegation contract')
                else:
                    raise DispatchError('Unsupported SDK envelope')
                await shutdown_if_done()
        parser.finish()
        if not shutdown_ack:
            raise DispatchError('Runtime EOF before successful shutdown response')

    params = {'cwd': str(root), 'provider': PROVIDER, 'model': MODEL, 'maxTokens': max_tokens}
    control_task = asyncio.create_task(steer_from_file()) if control_file is not None else None
    consume_task = None
    try:
        await send(1, 'initialize', params)
        consume_task = asyncio.create_task(consume())
        if control_task is not None:
            done, _ = await asyncio.wait((consume_task, control_task), return_when=asyncio.FIRST_COMPLETED)
            if control_task in done:
                await control_task
                if consume_task not in done:
                    raise DispatchError('Steering reader stopped before runtime completion')
        await consume_task
    except (DispatchError, KeyError, TypeError, ValueError, OSError) as exc:
        supervisor_errors.append(Diagnostics.redact(f'{type(exc).__name__}: {exc}')[:512])
    finally:
        if consume_task is not None:
            if not consume_task.done():
                consume_task.cancel()
            outcomes = await asyncio.gather(consume_task, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, Exception):
                    description = Diagnostics.redact(f'{type(outcome).__name__}: {outcome}')[:512]
                    if description not in supervisor_errors:
                        supervisor_errors.append(description)
        if control_task is not None:
            control_task.cancel()
            outcomes = await asyncio.gather(control_task, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, Exception):
                    description = Diagnostics.redact(f'{type(outcome).__name__}: {outcome}')[:512]
                    if description not in supervisor_errors:
                        supervisor_errors.append(description)
        stdout_drain_task = asyncio.create_task(drain_remaining_stdout())
        process.stdin.close()
        if supervisor_errors and process.returncode is None:
            process.terminate()
        try:
            await process.stdin.wait_closed()
        except OSError as exc:
            supervisor_errors.append(f'Stdin close failed: {type(exc).__name__}')
        await process.wait()
        for name, outcome in zip(('Stdout', 'Stderr'), await asyncio.gather(
                stdout_drain_task, stderr_task, return_exceptions=True)):
            if isinstance(outcome, Exception):
                supervisor_errors.append(f'{name} drain failed: {type(outcome).__name__}')
    if process.returncode != 0:
        supervisor_errors.append(f'Runtime exited with status {process.returncode}')
    for state in states.values():
        if state.terminal is None:
            state.terminal = {'kind': 'eof'}
        if state.queued_messages:
            state.failures.append(f'Runtime ended with {len(state.queued_messages)} admitted prompts unconsumed')
        if not state.headers:
            state.failures.append('No actual request header received')
        state.failures.extend(supervisor_errors)
    results = [state.compact() for state, _ in prepared]
    return {'tasks': results, 'runtime_exit': process.returncode,
            'supervisor_errors': supervisor_errors, 'stderr': diagnostics.result()}, int(any(not item['success'] for item in results))




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--brief', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS, help='Shared advertised output cap (default 131072; live acceptance unverified)')
    parser.add_argument('--resume-result', type=Path, action='append', help='Resume each matching assigned session from its public result; repeat per task')
    parser.add_argument('--resume-events', type=Path, action='append',
                        help='Read source correlation metadata from an assigned resumed session JSONL or JSONL.zstd; repeat per session')
    parser.add_argument('--control-file', type=Path, help='Append-only JSONL {task,text} for supported live same-session follow-ups')
    args = parser.parse_args()
    if not args.tasks or not args.brief or not args.output_dir:
        parser.error('--tasks, --brief and --output-dir are required')
    if args.max_tokens is not None and not 0 < args.max_tokens <= 2**53 - 1:
        parser.error('--max-tokens must be a positive safe integer')
    try:
        prepared = load_tasks(args.tasks, args.root, args.brief.read_text())
        if args.resume_result is not None:
            previous = [json.loads(path.read_text()) for path in args.resume_result]
            by_name = {item['name']: item for item in previous}
            if len(by_name) != len(previous) or set(by_name) != {state.name for state, _ in prepared}:
                raise DispatchError('Resume requires one public result for every matching assigned task')
            for state, _ in prepared:
                session_id = str(uuid.UUID(by_name[state.name]['session_id']))
                state.session_id = state.resume_session_id = session_id
                print(json.dumps({'event': 'resume_requested', 'task': state.name,
                                  'session_id': session_id}), flush=True)
        if args.resume_events is not None:
            states = {state.session_id: state for state, _ in prepared}
            for path in args.resume_events:
                load_resume_events(path, states)
        paths = [args.output_dir / f'{state.name}_result.json' for state, _ in prepared]
        if any(path.exists() for path in paths) or (args.output_dir / 'dispatch_result.json').exists():
            raise DispatchError('Output files already exist; choose a fresh output directory')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result, status = asyncio.run(run(prepared, args.root, args.max_tokens, control_file=args.control_file))
        for path, task in zip(paths, result['tasks']):
            path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + '\n')
        (args.output_dir / 'dispatch_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'output_dir': str(args.output_dir), 'runtime_exit': result['runtime_exit'],
                          'tasks': [{'name': task['name'], 'success': task['success'], 'terminal': task['terminal']} for task in result['tasks']]}))
        return status
    except (DispatchError, OSError, ValueError) as exc:
        print(Diagnostics.redact(f'Dispatch failed: {exc}')[:512], file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

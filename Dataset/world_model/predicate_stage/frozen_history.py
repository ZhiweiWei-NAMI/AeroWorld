"""Reuse actual frozen graph encodings across fitting and ablation exports."""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import pickle

import torch

from .train_predicate import causal_window_cache_path


def tensor_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tensor_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(tensor_tree(item, device) for item in value)
    return value


class FrozenHistoryCache:
    """Only graph output is cached; Qwen prefixes always use current LoRA weights.

    The key includes authentic source stamps/reader code, all graph weights,
    scales, graph code and runtime. File locks avoid repeating a costly history
    encoding when several ablation processes request the same window.
    """
    def __init__(self, graph, config, source_cache, directory, *, pooled_only=False):
        self.graph, self.config, self.source_cache = graph, config, source_cache
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.versions = None
        self.hits = self.misses = 0
        self.future_hits = self.future_misses = 0
        self.pooled_only = pooled_only
        self.pooled_hits = self.pooled_misses = 0

    def pooled(self, index, feedbacks, history_tokens, max_tokens, builder):
        """Small, source/weight-keyed graph prefixes; never cache Qwen states.

        Individual future prefixes allow K=1/2/8 to reuse the same draws. Raw
        full-graph slots and continuation state need not be written to disk.
        """
        from . import two_stage
        signature = self.fingerprint(index)
        code = hashlib.sha256(Path(two_stage.__file__).read_bytes()).hexdigest()
        base = hashlib.sha256(repr((signature, code, history_tokens, max_tokens)).encode()).hexdigest()
        history_path = self.directory / f"pooled_history_{base}.pt"
        paths = [] if feedbacks is None else [self.directory / ("pooled_future_"+hashlib.sha256(
            base.encode()+pickle.dumps(feedback)).hexdigest()+".pt") for feedback in feedbacks]
        if history_path.exists() and all(path.exists() for path in paths):
            history, diagnostic = torch.load(history_path, map_location="cpu", weights_only=False)
            self.pooled_hits += 1
            if feedbacks is None:
                return None, tensor_tree(history, self.graph.device), [{**diagnostic, "pooled_graph_cache_hit": True}]
            saved = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
            return (tensor_tree(history, self.graph.device),
                    [tensor_tree(prefix, self.graph.device) for prefix, _ in saved],
                    [{**d, "pooled_graph_cache_hit": True} for _, d in saved])
        result = builder()
        history, futures, diagnostics = result
        actual_history = futures if history is None else history
        temporary = history_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save((tensor_tree(actual_history, "cpu"), diagnostics[0]), temporary)
        os.replace(temporary, history_path)
        if history is not None:
            for path, prefix, diagnostic in zip(paths, futures, diagnostics):
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                torch.save((tensor_tree(prefix, "cpu"), diagnostic), temporary)
                os.replace(temporary, path)
        self.pooled_misses += 1
        return result

    def fingerprint(self, index):
        if any(parameter.requires_grad for parameter in self.graph.parameters()) or torch.is_grad_enabled():
            raise ValueError("history caching requires an explicitly frozen graph and no-grad encoding")
        versions = tuple(parameter._version for parameter in self.graph.parameters())
        if versions != self.versions:
            digest = hashlib.sha256()
            for name, tensor in self.graph.state_dict().items():
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            digest.update(Path(__file__).read_bytes())
            digest.update(Path(__file__).parents[1].joinpath("model/causal_input.py").read_bytes())
            digest.update(pickle.dumps(self.graph.observation_scales))
            digest.update(str((self.graph.mode, torch.__version__, torch.version.cuda,
                               self.graph.device.type, torch.backends.cuda.matmul.allow_tf32)).encode())
            self.graph_signature = digest.hexdigest()
            self.versions = versions
        source, _ = causal_window_cache_path((index.episode, index.cutoff), self.config, self.source_cache)
        digest = hashlib.sha256(("frozen-history-v1:"+source.stem+":"+self.graph_signature).encode())
        digest.update(pickle.dumps(index.scales))
        return digest.hexdigest()

    def load(self, index):
        signature = self.fingerprint(index)
        if self.pooled_only:
            self.misses += 1
            return self.graph.encode_history(index, None, reconstruct=False, return_state=True), False
        path = self.directory / f"{signature}.pt"
        with path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved["signature"] != signature or saved["window"] != (index.episode, index.cutoff):
                    raise ValueError("frozen history metadata does not match its source")
                result = tensor_tree(saved["history"], self.graph.device)
                state = result["continuation_state"]
                state.update(owner=id(self.graph), index=id(index), parameter_versions=self.versions)
                self.hits += 1
                return result, True
            result = self.graph.encode_history(index, None, reconstruct=False, return_state=True)
            cpu = tensor_tree(result, "cpu")
            # Object identities are process-local and rebound only after the
            # complete source/weight fingerprint has matched on a cache read.
            state = cpu["continuation_state"]
            for name in ("owner", "index", "parameter_versions"):
                state.pop(name)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save({"signature": signature, "window": (index.episode, index.cutoff), "history": cpu}, temporary)
            os.replace(temporary, path)
            self.misses += 1
            return result, False

    def load_future(self, index, feedback, state):
        """Cache frozen graph computation for a declared fixed Monte Carlo draw.

        The exact feedback, authentic source and graph weights key every entry.
        Qwen activations and learned predicate outputs are never cached here.
        """
        if self.pooled_only:
            self.future_misses += 1
            return self.graph.encode_future(index, feedback, state)
        signature = self.fingerprint(index)
        signature = hashlib.sha256(signature.encode()+pickle.dumps(feedback)).hexdigest()
        path = self.directory / f"future_{signature}.pt"
        if path.exists():
            self.future_hits += 1
            return tensor_tree(torch.load(path, map_location="cpu", weights_only=False), self.graph.device)
        result = self.graph.encode_future(index, feedback, state)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(tensor_tree(result, "cpu"), temporary)
        os.replace(temporary, path)
        self.future_misses += 1
        return result

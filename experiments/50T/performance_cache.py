"""Exact, bounded caches for the 50-turbine experiments.

No training budgets, objective formulas or random draws are changed. Persistent
baseline reuse replays the original post-baseline RNG state and records its
provenance; cached timings are not new independent measurements.
"""

import ast
import base64
import copy
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import textwrap
import time
from collections import OrderedDict, defaultdict

import numpy as np
from shapely.geometry import LineString, Point


CACHE_SCHEMA = 1
_STATS = defaultdict(lambda: {"hits": 0, "misses": 0, "evictions": 0,
                              "compute_seconds": 0.0})
_GLOBAL_CACHES = {}
_BASELINE_NAMES = (
    "JensenWakeModel", "EnhancedCFDModel", "_BaseOptimizer",
    "WhaleOptimizer", "GreyWolfOptimizer", "GeneticAlgorithm",
    "ParticleSwarmOptimizer", "SimulatedAnnealing", "BayesianOptimizer",
    "_frandsen_ti",
)
_LAYOUT_OUTPUTS = ("final_power_cvar", "final_power_std", "final_topology_feedback")


def caches_enabled():
    return os.environ.get("GNN_PPO_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _freeze(value):
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, np.generic):
        return ("scalar", value.dtype.str, value.tobytes())
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(val) for val in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(val) for val in value)
    if hasattr(value, "wkb"):
        return ("geometry", value.wkb)
    return value


def _settings(namespace):
    settings = {}
    for name in ("EnvironmentConfig", "ConstraintConfig", "LifecycleConfig",
                 "RobustConfig", "ReliabilityConfig"):
        cls = namespace.get(name)
        if cls is not None:
            settings[name] = {k: v for k, v in vars(cls).items()
                              if not k.startswith("_") and not callable(v)}
    settings["large_case"] = {k: v for k, v in namespace.items()
                              if k.startswith("LARGE_CASE_") and not callable(v)}
    return _freeze(settings)


def _model_key(model):
    return _freeze({k: v for k, v in vars(model).items()
                    if not k.startswith("_performance_")
                    and k not in ("n_calls", "total_t")})


def _store(owner, name):
    stores = owner.__dict__.setdefault("_performance_caches", {})
    return stores.setdefault(name, OrderedDict())


def _lookup(store, name, key):
    if key in store:
        _STATS[name]["hits"] += 1
        store.move_to_end(key)
        return True, copy.deepcopy(store[key])
    _STATS[name]["misses"] += 1
    return False, None


def _put(store, name, key, value, capacity):
    store[key] = copy.deepcopy(value)
    store.move_to_end(key)
    while len(store) > capacity:
        store.popitem(last=False)
        _STATS[name]["evictions"] += 1


def layout_cache_key(optimizer, layout, namespace):
    # Exact coordinates are essential: 10 m rounding aliases distinct layouts.
    return (_freeze(np.asarray(layout)), optimizer.num_turbines, optimizer.R,
            _freeze(optimizer.scenarios), _model_key(optimizer.cfd_model),
            _settings(namespace))


def cached_layout_objectives(fn):
    @functools.wraps(fn)
    def wrapped(self, layout):
        if not caches_enabled():
            return fn(self, layout)
        name = "layout_objectives"
        key = layout_cache_key(self, layout, fn.__globals__)
        store = _store(self, name)
        hit, entry = _lookup(store, name, key)
        if hit:
            result, outputs = entry
            for attr, val in outputs.items():
                setattr(self, attr, val)
            return result
        started = time.perf_counter()
        result = fn(self, layout)
        _STATS[name]["compute_seconds"] += time.perf_counter() - started
        # Early infeasible returns do not consistently assign the output fields.
        # Re-evaluate those instead of restoring another layout's stale metrics.
        if bool(result[-1]):
            outputs = {attr: getattr(self, attr) for attr in _LAYOUT_OUTPUTS
                       if hasattr(self, attr)}
            _put(store, name, key, (result, outputs), 4096)
        return result
    return wrapped


def cached_wake_evaluation(fn):
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not caches_enabled():
            return fn(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        params = {k: v for k, v in bound.arguments.items() if k != "self"}
        key = (_freeze(params), _model_key(self), _settings(fn.__globals__))
        name = "wake_with_ti"
        store = _store(self, name)
        hit, result = _lookup(store, name, key)
        if hit:
            return result
        started = time.perf_counter()
        result = fn(self, *args, **kwargs)
        _STATS[name]["compute_seconds"] += time.perf_counter() - started
        _put(store, name, key, result, 64)
        return result
    return wrapped


def _environment_key(env, scope, namespace):
    positions = (_freeze(env.node_positions), env.num_nodes, env.num_turbines)
    if scope == "obstacles":
        return _freeze(namespace["EnvironmentConfig"].FORBIDDEN_ZONES)
    if scope == "mst":
        return positions, namespace["EnvironmentConfig"].MAX_CABLE_LENGTH
    key = (positions, _settings(namespace))
    if scope == "geometry":
        return key, env.wind_direction
    key = (key, _freeze(env.adjacency_matrix))
    if scope == "topology":
        return key
    if scope != "metrics":
        raise ValueError("Unknown cache scope: " + scope)
    return (key, _freeze(env.turbine_positions), env.wind_speed, env.wind_direction, _freeze(env.yaw_angles),
            _freeze(env.wake_cache), _model_key(env._cfd_model))


def cached_environment_method(scope):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            if not caches_enabled():
                return fn(self, *args, **kwargs)
            name = "env." + fn.__name__
            arg_key = (_freeze(args), _freeze(kwargs))
            key = (_environment_key(self, scope, fn.__globals__), arg_key)
            store = _store(self, name)
            valid = scope != "metrics" or self.wake_cache.get("is_valid", False)
            if valid:
                hit, result = _lookup(store, name, key)
                if hit:
                    return result
            else:
                _STATS[name]["misses"] += 1
            started = time.perf_counter()
            result = fn(self, *args, **kwargs)
            _STATS[name]["compute_seconds"] += time.perf_counter() - started
            if scope != "metrics" or self.wake_cache.get("is_valid", False):
                # Some metrics refresh the wake on entry. Store the refreshed key.
                key = (_environment_key(self, scope, fn.__globals__), arg_key)
                _put(store, name, key, result, 8 if scope != "obstacles" else 2048)
            return result
        return wrapped
    return decorate


def _geometric_value(name, key, compute, capacity):
    if not caches_enabled():
        return compute()
    store = _GLOBAL_CACHES.setdefault(name, OrderedDict())
    if key in store:
        _STATS[name]["hits"] += 1
        store.move_to_end(key)
        return store[key]
    _STATS[name]["misses"] += 1
    result = compute()
    store[key] = result
    if len(store) > capacity:
        store.popitem(last=False)
        _STATS[name]["evictions"] += 1
    return result


def cached_line(start, end):
    key = (tuple(start), tuple(end))
    return _geometric_value("geometry.lines", key,
                            lambda: LineString([start, end]), 8192)


def cached_obstacle_polygon(center, radius):
    key = (tuple(center), radius)
    return _geometric_value("geometry.obstacles", key,
                            lambda: Point(center).buffer(radius), 128)


def cached_line_conflict(first, second):
    key = tuple(sorted((first.wkb, second.wkb)))
    return _geometric_value("geometry.intersections", key,
                            lambda: first.intersects(second) and not first.touches(second),
                            32768)


def _pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be persisted in a baseline cache")
        return {"kind": "array", "dtype": value.dtype.str, "shape": list(value.shape),
                "data": base64.b64encode(value.tobytes()).decode("ascii")}
    if isinstance(value, np.generic):
        return {"kind": "scalar", "value": _pack(np.asarray(value))}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_pack(v) for v in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_pack(v) for v in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[k, _pack(v)] for k, v in value.items()]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported cache value: " + type(value).__name__)


def _unpack(value):
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "array":
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Object dtype in baseline cache")
        return np.frombuffer(base64.b64decode(value["data"], validate=True),
                             dtype=dtype).reshape(value["shape"]).copy()
    if kind == "scalar":
        return _unpack(value["value"])[()]
    if kind == "tuple":
        return tuple(_unpack(v) for v in value["items"])
    if kind == "list":
        return [_unpack(v) for v in value["items"]]
    if kind == "dict":
        return {k: _unpack(v) for k, v in value["items"]}
    raise ValueError("Unknown baseline cache value type")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(_pack(value))).hexdigest()


def _rng_snapshot():
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state().cpu().numpy().copy()
        # Do not initialise CUDA merely to cache a CPU-only preliminary phase.
        if torch.cuda.is_initialized():
            state["torch_cuda"] = [s.cpu().numpy().copy()
                                     for s in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch = sys.modules.get("torch")
    if "torch_cpu" in state:
        torch.set_rng_state(torch.from_numpy(state["torch_cpu"].copy()))
    if "torch_cuda" in state:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from cached baseline")
        torch.cuda.set_rng_state_all([torch.from_numpy(s.copy()) for s in state["torch_cuda"]])


def _normalised_source(obj):
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(obj))))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.decorator_list = []
    return tree, ast.dump(tree, include_attributes=False)


def baseline_identity(fn, args, kwargs, rng):
    fn = inspect.unwrap(fn)
    namespace = fn.__globals__
    objects = {"baseline": inspect.unwrap(fn)}
    objects.update({name: namespace[name] for name in _BASELINE_NAMES if name in namespace})
    sources, config, constants = {}, {}, {}
    for name, obj in objects.items():
        tree, source = _normalised_source(obj)
        sources[name] = source
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                cls_name = node.value.id
                cls = namespace.get(cls_name)
                if cls_name.endswith("Config") and cls is not None:
                    config[cls_name + "." + node.attr] = getattr(cls, node.attr)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                val = namespace.get(node.id)
                if node.id.isupper() and isinstance(val, (str, int, float, tuple, list, dict, np.ndarray)):
                    constants[node.id] = val
    root = Path(namespace["__file__"]).resolve().parent
    files = [Path(__file__).resolve(), root / "large_case_config.py"]
    if (root / "case_data").is_dir():
        files.extend(sorted(p for p in (root / "case_data").rglob("*") if p.is_file()))
    file_hashes = {str(p.relative_to(root)) if p.is_relative_to(root) else p.name:
                   hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    versions = {name: getattr(sys.modules.get(name), "__version__", None)
                for name in ("numpy", "scipy", "sklearn", "networkx", "shapely", "torch")}
    torch = sys.modules.get("torch")
    runtime = {"python": platform.python_version(), "system": platform.system(),
               "processor": platform.processor(),
               "machine": platform.machine(), "versions": versions,
               "threads": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")}}
    if torch is not None:
        runtime["torch_threads"] = torch.get_num_threads()
        runtime["torch_interop_threads"] = torch.get_num_interop_threads()
    identity = {"schema": CACHE_SCHEMA, "args": args, "kwargs": kwargs,
                "seed": namespace.get("RANDOM_SEED"), "entry_rng": rng,
                "sources": sources, "config": config, "constants": constants,
                "files": file_hashes, "runtime": runtime}
    return _digest(identity)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as out:
            temporary = Path(out.name)
            json.dump(value, out, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _provenance(namespace, record):
    target = namespace.get("PLOT_DATA_DIR")
    if target:
        try:
            _atomic_json(Path(target) / "baseline_cache.json", record)
        except OSError as exc:
            namespace["LOG"].warning("Cannot write baseline provenance: %s", exc)


def cached_baseline_comparison(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        namespace = fn.__globals__
        log = namespace["LOG"]
        started = time.perf_counter()
        enabled = caches_enabled() and os.environ.get("GNN_PPO_BASELINE_CACHE", "1") != "0"
        record = {"schema": CACHE_SCHEMA, "status": "disabled", "hit": False,
                  "timings_reused": False, "random_seed": namespace.get("RANDOM_SEED")}
        path, entry_rng, key = None, None, None
        if enabled:
            try:
                entry_rng = _rng_snapshot()
                key = baseline_identity(fn, args, kwargs, entry_rng)
                root = Path(namespace["__file__"]).resolve().parent
                cache_dir = Path(os.environ.get("GNN_PPO_CACHE_DIR", str(root / ".performance_cache")))
                path = cache_dir / "baselines" / (key + ".json")
                record.update(status="miss", key=key, cache_file=str(path))
                if path.is_file():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    body = payload["body"]
                    if (payload["schema"] != CACHE_SCHEMA or payload["key"] != key
                            or hashlib.sha256(_json_bytes(body)).hexdigest() != payload["sha256"]):
                        raise ValueError("Baseline cache checksum/schema mismatch")
                    result = _unpack(body["result"])
                    if not isinstance(result, tuple) or len(result) != 4:
                        raise ValueError("Invalid baseline result structure")
                    _restore_rng(_unpack(body["exit_rng"]))
                    record.update(status="hit", hit=True, timings_reused=True,
                                  source=body["source"],
                                  original_compute_seconds=body["compute_seconds"],
                                  elapsed_seconds=time.perf_counter() - started)
                    _STATS["baseline"]["hits"] += 1
                    log.info("[CACHE] Baseline HIT %s; source=%s; replayed post-baseline RNG. "
                             "Baseline histories/timings are reused, not independent runs.",
                             key[:12], body["source"]["script"])
                    _provenance(namespace, record)
                    return result
                log.info("[CACHE] Baseline MISS %s; computing the unchanged 12 comparisons.", key[:12])
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                if entry_rng is not None:
                    _restore_rng(entry_rng)
                record.update(status="recomputed", cache_error=str(exc))
                log.warning("[CACHE] Baseline unavailable; recomputing safely: %s", exc)
        _STATS["baseline"]["misses"] += 1
        compute_start = time.perf_counter()
        result = fn(*args, **kwargs)
        compute_seconds = time.perf_counter() - compute_start
        _STATS["baseline"]["compute_seconds"] += compute_seconds
        if enabled and path is not None and key is not None:
            try:
                source = {"script": Path(namespace["__file__"]).name,
                          "results_dir": str(Path(namespace["RESULTS_DIR"]).resolve()),
                          "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "host": platform.node()}
                body = {"result": _pack(result), "exit_rng": _pack(_rng_snapshot()),
                        "source": source, "compute_seconds": compute_seconds}
                _atomic_json(path, {"schema": CACHE_SCHEMA, "key": key, "body": body,
                                    "sha256": hashlib.sha256(_json_bytes(body)).hexdigest()})
                record.update(source=source, original_compute_seconds=compute_seconds)
            except (OSError, ValueError, TypeError) as exc:
                record["write_error"] = str(exc)
                log.warning("[CACHE] Cannot persist baseline; training will continue: %s", exc)
        record["elapsed_seconds"] = time.perf_counter() - started
        _provenance(namespace, record)
        return result
    return wrapped


def cache_statistics():
    return {"schema": CACHE_SCHEMA, "enabled": caches_enabled(),
            "counters": copy.deepcopy(dict(_STATS)),
            "note": "Hits skip exact repeat evaluations, not new training samples. "
                    "compute_seconds counts cache misses only, not estimated saved time."}


def cache_reported_main(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        namespace = fn.__globals__
        namespace["LOG"].info("[CACHE] Exact layout/wake/geometry caches enabled=%s", caches_enabled())
        try:
            return fn(*args, **kwargs)
        finally:
            try:
                path = Path(namespace["PLOT_DATA_DIR"]) / "performance_cache_stats.json"
                _atomic_json(path, cache_statistics())
                namespace["LOG"].info("[CACHE] Statistics saved: %s", path)
            except OSError as exc:
                namespace["LOG"].warning("Cannot write cache statistics: %s", exc)
    return wrapped

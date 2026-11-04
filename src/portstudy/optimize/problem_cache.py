"""Compiled cvxpy problems keyed by their constant data, with the expected return as a
parameter, so repeated solves (resampled draws, shuffles, many sources per week) skip
canonicalization. The key always includes the ticker tuple, so a universe change misses."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np

PROBLEM_CACHE_SIZE = 16


@dataclass
class CachedProblem:
    problem: cp.Problem
    weights: cp.Variable
    mu: cp.Parameter
    aux: dict[str, Any] = field(default_factory=dict)


_CACHE: OrderedDict[Hashable, CachedProblem] = OrderedDict()
_STATS = {"hits": 0, "misses": 0}


def matrix_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def cached_problem(key: Hashable, builder: Callable[[], CachedProblem]) -> CachedProblem:
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        _STATS["hits"] += 1
        return hit
    built = builder()
    _CACHE[key] = built
    _STATS["misses"] += 1
    while len(_CACHE) > PROBLEM_CACHE_SIZE:
        _CACHE.popitem(last=False)
    return built


def clear_problem_cache() -> None:
    _CACHE.clear()
    _STATS["hits"] = 0
    _STATS["misses"] = 0


def problem_cache_stats() -> dict[str, int]:
    return {"size": len(_CACHE), **_STATS}

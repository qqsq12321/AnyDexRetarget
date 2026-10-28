"""Exact affine elimination of scalar URDF mimic joints.

No post-FK clipping: limits on every dependent joint are pulled back onto
its independent ancestor. This also handles negative gains and nested chains.
"""
from __future__ import annotations

import numpy as np


class MimicMap:
    def __init__(self, num_joints: int, relations: dict):
        self.independent_indices = np.array(
            [j for j in range(num_joints) if j not in relations], dtype=np.int64
        )
        if not len(self.independent_indices):
            raise ValueError("Mimic graph has no independent joint (possible cycle)")
        self.matrix = np.zeros((num_joints, len(self.independent_indices)))
        self.offset = np.zeros(num_joints)
        columns = {int(j): i for i, j in enumerate(self.independent_indices)}
        state = np.zeros(num_joints, dtype=np.int8)

        def resolve(j):
            if not 0 <= j < num_joints:
                raise ValueError(f"Mimic joint index out of range: {j}")
            if state[j] == 1:
                raise ValueError(f"Cycle in mimic graph at joint {j}")
            if state[j] == 2:
                return
            state[j] = 1
            if j in relations:
                source, multiplier, offset = relations[j]
                if not np.all(np.isfinite([multiplier, offset])):
                    raise ValueError("Mimic multiplier and offset must be finite")
                resolve(source)
                self.matrix[j] = multiplier * self.matrix[source]
                self.offset[j] = multiplier * self.offset[source] + offset
            else:
                self.matrix[j, columns[j]] = 1.0
            state[j] = 2

        for j in relations:
            if not 0 <= j < num_joints:
                raise ValueError(f"Mimic joint index out of range: {j}")
        for j in range(num_joints):
            resolve(j)
        if not np.all(np.isfinite(self.matrix)) or not np.all(np.isfinite(self.offset)):
            raise ValueError("Mimic chain overflows")

    def independent_bounds(self, lower, upper):
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if lower.shape != self.offset.shape or upper.shape != self.offset.shape:
            raise ValueError("Joint bounds must match full joint dimension")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("Invalid joint bounds")
        lo = np.full(self.matrix.shape[1], -np.inf)
        hi = np.full(self.matrix.shape[1], np.inf)
        for j, row in enumerate(self.matrix):
            nz = np.flatnonzero(row)
            if not len(nz):
                if not lower[j] <= self.offset[j] <= upper[j]:
                    raise ValueError(f"Constant mimic joint {j} is outside its limits")
                continue
            k = nz[0]  # URDF mimic has a single source, even after composition.
            a = (lower[j] - self.offset[j]) / row[k]
            b = (upper[j] - self.offset[j]) / row[k]
            lo[k] = max(lo[k], min(a, b))
            hi[k] = min(hi[k], max(a, b))
        if np.any(lo > hi):
            raise ValueError("Mimic relationships and joint limits have no feasible intersection")
        return lo, hi

    def expand(self, independent):
        x = np.asarray(independent, dtype=np.float64)
        if x.shape != (self.matrix.shape[1],):
            raise ValueError("Incorrect independent joint dimension")
        return self.matrix @ x + self.offset

    def gradient(self, full_gradient):
        return self.matrix.T @ np.asarray(full_gradient, dtype=np.float64)

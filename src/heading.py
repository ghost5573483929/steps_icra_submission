"""Central-difference heading for contiguous canonical tracks.

The canonical schema stores a per-row ``heading`` in radians. Adapters whose tracks
are contiguous on the canonical lattice (UCY, Grand Central) share this convention:

* interior rows use the central difference ``xy[i + 1] - xy[i - 1]``;
* the first and last row use the one-sided difference against their single neighbour;
* a displacement at or below :data:`HEADING_SPEED_EPSILON` world units leaves the
  heading undefined (``NaN``), because ``arctan2`` of a stationary step has no direction.

Other adapters either guard contiguity themselves before differencing (madras) or take
heading from a source-provided column (STEPS, ETH) instead of using this helper.
"""

from __future__ import annotations

import numpy as np


# Displacement floor, in canonical world units, below which heading is undefined.
# A fixed convention shared by every adapter; do not retune it per dataset.
HEADING_SPEED_EPSILON = 1e-8


def central_difference_heading(xy: np.ndarray) -> np.ndarray:
    """Return the heading in radians for each row of a contiguous track.

    ``xy`` is an ``(n, 2)`` array of positions ordered by step, with no gaps on
    the canonical lattice.  The result is a length-``n`` float array holding
    ``arctan2(dy, dx)`` in ``[-pi, pi]``, or ``NaN`` where the differencing
    stencil yields a displacement of at most :data:`HEADING_SPEED_EPSILON`.

    A single-position track has no neighbour to difference against, so its
    heading is ``NaN`` by the same convention.  An empty track is rejected.
    """
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"expected an (n, 2) array of positions, got shape {points.shape}")
    if len(points) == 0:
        raise ValueError("cannot derive heading for a track with no positions")
    if len(points) == 1:
        return np.array([np.nan])
    delta = np.empty_like(points)
    delta[0] = points[1] - points[0]
    delta[-1] = points[-1] - points[-2]
    if len(points) > 2:
        delta[1:-1] = points[2:] - points[:-2]
    speed = np.linalg.norm(delta, axis=1)
    return np.where(speed > HEADING_SPEED_EPSILON, np.arctan2(delta[:, 1], delta[:, 0]), np.nan)

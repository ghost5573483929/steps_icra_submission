from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class EarlyStopping:
    """Epoch-level early stopping for metrics where lower is better."""

    patience: int = 10
    min_delta: float = 1e-4
    best: float = float("inf")
    bad_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        improved = math.isfinite(value) and value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        stopped = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, stopped


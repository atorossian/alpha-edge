# backtest/goals.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple


Goals3 = Tuple[float, float, float]

def goals3_from_goalpath(gp: GoalPath) -> Goals3:
    """
    Returns a 3-goal window starting at gp.idx.

    Guarantees:
    - goals3[0] == gp.main_goal
    - padded by repeating the last goal if near the end
    """
    g = gp.goals
    i = int(gp.idx)

    def at(k: int) -> float:
        return float(g[min(k, len(g) - 1)])

    return (at(i), at(i + 1), at(i + 2))


@dataclass
class GoalPath:
    goals: List[float]
    idx: int = 0
    hit_streak: int = 0
    confirm_steps: int = 1
    buffer: float = 0.0

    def __post_init__(self):
        self.goals = sorted([float(g) for g in self.goals if g is not None])
        if not self.goals:
            raise ValueError("GoalPath.goals must be non-empty")
        self.idx = max(0, min(int(self.idx), len(self.goals) - 1))
        self.hit_streak = int(self.hit_streak)
        self.confirm_steps = max(1, int(self.confirm_steps))
        self.buffer = float(self.buffer)

    @property
    def main_goal(self) -> float:
        return float(self.goals[self.idx])

    @property
    def next_goal(self) -> Optional[float]:
        return float(self.goals[self.idx + 1]) if self.idx + 1 < len(self.goals) else None

    def update(self, equity: float) -> dict:
        equity = float(equity)
        g = self.main_goal
        thresh = g * (1.0 + self.buffer)

        reached = equity >= thresh
        self.hit_streak = (self.hit_streak + 1) if reached else 0

        prev_idx = self.idx
        if self.hit_streak >= self.confirm_steps:
            # advance across multiple goals if equity jumped
            while self.idx + 1 < len(self.goals) and equity >= self.goals[self.idx] * (1.0 + self.buffer):
                self.idx += 1
                if equity < self.goals[self.idx] * (1.0 + self.buffer):
                    break
            self.hit_streak = 0

        return {
            "prev_goal": float(self.goals[prev_idx]),
            "new_goal": float(self.main_goal),
            "advanced": bool(self.idx != prev_idx),
            "equity": float(equity),
        }
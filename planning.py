from dataclasses import dataclass
from typing import List, Dict, Optional
from time import time
import torch

@dataclass
class Goal:
    description: str
    priority: float
    steps: List[str] = None
    progress: float = 0.0
    created_at: float = None
    status: str = "active"

    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.created_at is None:
            self.created_at = time()

class Planner:
    def __init__(self, self_model, future_system):
        self.self_model = self_model
        self.future_system = future_system
        self.goals: List[Goal] = []

    def form_goal(self, state, curiosity: float, threat: float, prediction_error: float) -> Optional[Goal]:
        # Правила формирования цели (без LLM)
        if curiosity > 0.7:
            desc = "Узнать больше о собеседнике"
            priority = curiosity
        elif threat > 0.5:
            desc = "Поддерживать разговор"
            priority = threat
        elif prediction_error > 0.6:
            desc = "Разобраться в неожиданном событии"
            priority = prediction_error
        elif state.valence < -0.3:
            desc = "Избежать негативных тем"
            priority = -state.valence
        else:
            return None

        # Простая декомпозиция правилами
        if "Узнать больше" in desc:
            steps = ["Задать вопрос о хобби", "Задать вопрос о работе", "Задать вопрос о планах"]
        elif "Поддерживать разговор" in desc:
            steps = ["Задать открытый вопрос", "Поделиться своим состоянием"]
        elif "Разобраться" in desc:
            steps = ["Задать уточняющий вопрос", "Поделиться своим мнением"]
        else:
            steps = ["Задать нейтральный вопрос"]

        return Goal(description=desc, priority=priority, steps=steps)

    def update_progress(self, goal: Goal, success: bool):
        delta = 0.3 if success else -0.1
        goal.progress = min(1.0, max(0.0, goal.progress + delta))
        if goal.progress >= 1.0:
            goal.status = "completed"
        elif goal.progress <= 0:
            goal.status = "failed"

    def get_current_action(self) -> Optional[str]:
        active_goals = [g for g in self.goals if g.status == "active"]
        if not active_goals:
            return None
        top_goal = max(active_goals, key=lambda g: g.priority)
        step_idx = min(int(len(top_goal.steps) * top_goal.progress), len(top_goal.steps) - 1)
        if step_idx < len(top_goal.steps):
            return top_goal.steps[step_idx]
        return None

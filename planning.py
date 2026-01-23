from dataclasses import dataclass
from typing import List, Optional
from time import time

@dataclass
class Goal:
    description: str
    priority: float
    steps: List[str]
    progress: float = 0.0
    created_at: float = None
    status: str = "active"

    def __post_init__(self):
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
        else:
            steps = ["Анализировать ситуацию", "Задать уточняющий вопрос"]

        return Goal(description=desc, priority=priority, steps=steps)

    def simulate_outcome(self, action: str) -> float:
        # Простая симуляция valence по действию (non-verbal reasoning)
        if "задать вопрос" in action.lower():
            return 0.3  # Положительный исход от любопытства
        elif "избежать" in action.lower():
            return 0.1
        return 0.0

    def choose_action(self) -> Optional[str]:
        if not self.goals:
            return None
        top_goal = max(self.goals, key=lambda g: g.priority)
        step_idx = int(len(top_goal.steps) * top_goal.progress)
        if step_idx < len(top_goal.steps):
            return top_goal.steps[step_idx]
        return None

    def update(self, success: bool):
        for goal in self.goals:
            goal.progress += 0.3 if success else -0.1
            if goal.progress >= 1.0:
                goal.status = "completed"
            elif goal.progress <= 0:
                goal.status = "failed"

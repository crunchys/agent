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
    def __init__(self, self_model, future_system, simulator=None):  # НОВОЕ: добавлен simulator
        self.self_model = self_model
        self.future_system = future_system
        self.simulator = simulator  # НОВОЕ
        self.goals: List[Goal] = []

    def form_goal(self, state, curiosity: float, threat: float, prediction_error: float) -> Optional[Goal]:
        # Правила формирования цели (без LLM) - БЕЗ ИЗМЕНЕНИЙ
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

        # Простая декомпозиция правилами - БЕЗ ИЗМЕНЕНИЙ
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
        elif goal.progress <= 0 and goal.created_at < (time() - 30):  # ИЗМЕНЕНО: дали больше времени
            # Цель фейлится только если прошло 30+ секунд И progress = 0
            goal.status = "failed"

    def get_current_action(self) -> Optional[str]:
        # БЕЗ ИЗМЕНЕНИЙ
        active_goals = [g for g in self.goals if g.status == "active"]
        if not active_goals:
            return None
        top_goal = max(active_goals, key=lambda g: g.priority)
        step_idx = min(int(len(top_goal.steps) * top_goal.progress), len(top_goal.steps) - 1)
        if step_idx < len(top_goal.steps):
            return top_goal.steps[step_idx]
        return None
    
    # ===== НОВОЕ: Методы для работы с симулятором =====
    
    def choose_best_action_simulated(self, state, available_actions: List[str]) -> Optional[str]:
        """
        НОВОЕ: Выбрать лучшее действие через симуляцию.
        Если симулятор не доступен - вернуть первое действие.
        """
        if not self.simulator or not available_actions:
            return available_actions[0] if available_actions else None
        
        # Преобразуем строки в Dict для симулятора
        actions_dict = [{"description": action, "type": "conversational"} for action in available_actions]
        
        # Сравниваем действия через симулятор
        ranked_actions = self.simulator.compare_actions(state, actions_dict)
        
        if ranked_actions:
            best = ranked_actions[0]
            print(f"[СИМУЛЯЦИЯ] Выбрано: '{best['action']['description']}'")
            print(f"  → Utility: {best['utility']:.3f}")
            print(f"  → Expected valence: {best['simulation']['expected_valence']:+.2f}")
            print(f"  → Risk: {best['simulation']['risk']:.2f}")
            return best['action']['description']
        
        return available_actions[0] if available_actions else None
    
    def simulate_goal_outcome(self, goal: Goal, state) -> Dict:
        """
        НОВОЕ: Симулировать исход всей цели (не только одного действия).
        Используется для приоритизации целей.
        """
        if not self.simulator or not goal.steps:
            return {"expected_valence": 0.0, "confidence": 0.0}
        
        # Симулируем первый шаг цели
        first_step = {"description": goal.steps[0], "type": "goal_step"}
        outcome = self.simulator.simulate_action(state, first_step, n_rollouts=2)
        
        return outcome
    
    def replan_if_needed(self, state) -> bool:
        """
        НОВОЕ: Проверить, нужно ли пересмотреть план на основе симуляций.
        Возвращает True если план был изменён.
        """
        if not self.simulator or not self.goals:
            return False
        
        active_goals = [g for g in self.goals if g.status == "active"]
        if not active_goals:
            return False
        
        # Симулируем исходы всех активных целей
        simulations = []
        for goal in active_goals:
            outcome = self.simulate_goal_outcome(goal, state)
            simulations.append({
                "goal": goal,
                "outcome": outcome
            })
        
        # Проверяем, не ухудшится ли состояние сильно
        current_best = max(active_goals, key=lambda g: g.priority)
        
        for sim in simulations:
            if sim["goal"] == current_best:
                # Если текущая лучшая цель приведёт к сильному негативу - сменить
                if sim["outcome"]["expected_valence"] < -0.4:
                    print(f"[REPLANNING] Цель '{current_best.description}' приведёт к негативу")
                    current_best.status = "cancelled"
                    return True
        
        return False

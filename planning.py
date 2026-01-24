from dataclasses import dataclass
from typing import List, Dict, Optional
from time import time
import torch
import random

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
    def __init__(self, self_model, future_system, simulator=None):
        self.self_model = self_model
        self.future_system = future_system
        self.simulator = simulator
        self.goals: List[Goal] = []
        
        # НОВОЕ: Словарь конфликтов мотиваций
        self.motivation_conflicts = {
            ("избегать", "узнать"): 0.7,
            ("поддерживать разговор", "избежать негативных"): 0.6,
            ("разобраться", "избегать"): 0.5,
        }

    def form_goal(self, state, curiosity: float, threat: float, prediction_error: float) -> Optional[Goal]:
        # Правила формирования цели
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
        elif goal.progress <= 0 and goal.created_at < (time() - 30):
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
    
    # ===== Методы для работы с симулятором =====
    
    def choose_best_action_simulated(self, state, available_actions: List[str]) -> Optional[str]:
        """Выбрать лучшее действие через симуляцию"""
        if not self.simulator or not available_actions:
            return available_actions[0] if available_actions else None
        
        actions_dict = [{"description": action, "type": "conversational"} for action in available_actions]
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
        """Симулировать исход всей цели"""
        if not self.simulator or not goal.steps:
            return {"expected_valence": 0.0, "confidence": 0.0}
        
        first_step = {"description": goal.steps[0], "type": "goal_step"}
        outcome = self.simulator.simulate_action(state, first_step, n_rollouts=2)
        
        return outcome
    
    def replan_if_needed(self, state) -> bool:
        """Проверить, нужно ли пересмотреть план на основе симуляций"""
        if not self.simulator or not self.goals:
            return False
        
        active_goals = [g for g in self.goals if g.status == "active"]
        if not active_goals:
            return False
        
        simulations = []
        for goal in active_goals:
            outcome = self.simulate_goal_outcome(goal, state)
            simulations.append({
                "goal": goal,
                "outcome": outcome
            })
        
        current_best = max(active_goals, key=lambda g: g.priority)
        
        for sim in simulations:
            if sim["goal"] == current_best:
                if sim["outcome"]["expected_valence"] < -0.4:
                    print(f"[REPLANNING] Цель '{current_best.description}' приведёт к негативу")
                    current_best.status = "cancelled"
                    return True
        
        return False
    
    # ===== НОВОЕ: Принудительное выполнение плана =====
    
    def enforce_action(self, response: str, current_action: str, state) -> str:
        """
        НОВОЕ: Принудительно выполнить план - проверить и дополнить ответ если нужно.
        Архитектурное влияние плана на ответ.
        """
        if not current_action or current_action == "нет":
            return response
        
        action_lower = current_action.lower()
        
        # 1. Если план - задать вопрос, но в ответе нет "?"
        if "вопрос" in action_lower and "?" not in response:
            # Извлечь тему вопроса
            topic = self._extract_topic(current_action)
            
            # Добавить вопрос
            questions = [
                f" А что насчёт {topic}?",
                f" Расскажи мне о {topic}?",
                f" Как у тебя с {topic}?",
                f" Что думаешь о {topic}?"
            ]
            question = random.choice(questions)
            
            print(f"[ПЛАН] Добавлен вопрос: '{question.strip()}'")
            return response.rstrip() + question
        
        # 2. Если план - поделиться состоянием, но нет упоминания эмоций
        if "поделиться" in action_lower and "состоянием" in action_lower:
            # Проверяем, есть ли эмоциональные слова
            emotion_words = ["чувствую", "ощущ", "настро", "состо", "эмоц", "волну", "рад", "груст", "тревож"]
            has_emotion = any(word in response.lower() for word in emotion_words)
            
            if not has_emotion:
                # Добавить упоминание состояния
                emotion_mentions = [
                    " Сейчас я чувствую себя довольно спокойно.",
                    " Немного задумчив, если честно.",
                    " В целом настроение хорошее.",
                ]
                mention = random.choice(emotion_mentions)
                
                print(f"[ПЛАН] Добавлено состояние: '{mention.strip()}'")
                return response.rstrip() + mention
        
        # 3. Если план - поделиться мнением, но ответ слишком нейтральный
        if "мнением" in action_lower or "мнение" in action_lower:
            # Проверяем, есть ли слова мнения
            opinion_words = ["думаю", "считаю", "мне кажется", "по-моему", "на мой взгляд", "уверен"]
            has_opinion = any(word in response.lower() for word in opinion_words)
            
            if not has_opinion and len(response) < 100:
                # Добавить фразу мнения
                opinion_phrases = [
                    " Я думаю, это интересная тема.",
                    " Мне кажется, стоит разобраться глубже.",
                    " По-моему, здесь есть над чем подумать.",
                ]
                phrase = random.choice(opinion_phrases)
                
                print(f"[ПЛАН] Добавлено мнение: '{phrase.strip()}'")
                return response.rstrip() + phrase
        
        return response
    
    def _extract_topic(self, action: str) -> str:
        """Извлечь тему из действия (например, 'Задать вопрос о хобби' → 'хобби')"""
        # Ищем паттерн "о X" или "про X"
        if " о " in action:
            topic = action.split(" о ")[-1].strip()
            return topic
        elif " про " in action:
            topic = action.split(" про ")[-1].strip()
            return topic
        
        # По умолчанию - общая тема
        return "тебе"
    
    # ===== НОВОЕ: Конфликт мотиваций =====
    
    def detect_motivation_conflict(self, state) -> float:
        """
        НОВОЕ: Определить конфликт мотиваций.
        Возвращает уровень конфликта (0.0-1.0).
        """
        if not self.goals or len(self.goals) < 2:
            return 0.0
        
        active_goals = [g for g in self.goals if g.status == "active"]
        if len(active_goals) < 2:
            return 0.0
        
        # Проверяем все пары целей на конфликт
        max_conflict = 0.0
        conflicting_pair = None
        
        for i, goal1 in enumerate(active_goals):
            for goal2 in active_goals[i+1:]:
                conflict_level = self._check_conflict(goal1.description, goal2.description)
                if conflict_level > max_conflict:
                    max_conflict = conflict_level
                    conflicting_pair = (goal1.description, goal2.description)
        
        if max_conflict > 0.5 and conflicting_pair:
            print(f"[КОНФЛИКТ] Противоречивые цели!")
            print(f"  → '{conflicting_pair[0]}' vs '{conflicting_pair[1]}'")
            print(f"  → Уровень конфликта: {max_conflict:.2f}")
        
        return max_conflict
    
    def _check_conflict(self, desc1: str, desc2: str) -> float:
        """Проверить две цели на конфликт"""
        desc1_lower = desc1.lower()
        desc2_lower = desc2.lower()
        
        for (pattern1, pattern2), conflict_level in self.motivation_conflicts.items():
            if pattern1 in desc1_lower and pattern2 in desc2_lower:
                return conflict_level
            if pattern2 in desc1_lower and pattern1 in desc2_lower:
                return conflict_level
        
        return 0.0

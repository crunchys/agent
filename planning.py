from dataclasses import dataclass
from typing import List, Dict, Optional
from time import time
import torch

@dataclass
class Goal:
    description: str
    priority: float  # 0.0 - 1.0
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
    def __init__(self, model, tokenizer, self_model):
        self.model = model
        self.tokenizer = tokenizer
        self.self_model = self_model
        self.goals: List[Goal] = []

    def form_goal(self, state, curiosity: float, threat: float) -> Optional[Goal]:
        prompt = (
            f"Состояние: arousal={state.arousal:.2f}, valence={state.valence:.2f}, threat={threat:.2f}, curiosity={curiosity:.2f}.\n"
            f"Мотивации: {', '.join(self.self_model.motivations)}.\n"
            "Сформулируй одну реалистичную цель для разговора с человеком (1 предложение). "
            "Или верни 'нет цели'."
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=60,
            temperature=0.8
        )
        goal_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
        
        if "нет цели" in goal_text.lower():
            return None
        
        priority = curiosity * 0.7 + threat * 0.2 + max(0, -state.valence) * 0.1
        return Goal(description=goal_text, priority=min(1.0, priority))

    def decompose_goal(self, goal: Goal) -> List[str]:
        prompt = (
            f"Цель: {goal.description}\n"
            "Разбей на 3–5 простых шагов в разговоре (например, 'задать вопрос о...'). Нумеруй."
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            temperature=0.7
        )
        steps_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
        steps = [s.strip() for s in steps_text.split('\n') if s.strip() and s[0].isdigit()]
        goal.steps = steps
        return steps

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

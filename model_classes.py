from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

@dataclass
class SelfModel:
    name: str = "Агент"
    traits: Dict[str, float] = None
    goals: List[str] = None
    lessons: List[str] = None
    motivations: List[str] = None
    self_evaluations: List[str] = None

    def __post_init__(self):
        if self.traits is None:
            self.traits = {'смелость': 0.5, 'любопытство': 0.5}
        if self.goals is None:
            self.goals = []
        if self.lessons is None:
            self.lessons = []
        if self.motivations is None:
            self.motivations = ['избегать неудач', 'искать знания', 'поддерживать существование']
        if self.self_evaluations is None:
            self.self_evaluations = []

    def reflect(self, state, memory, new_lesson: Optional[str] = None):
        # Фильтруем только события типа "experience", чтобы избежать KeyError
        recent_experiences = [e for e in memory.recent(5) if e.get("type") == "experience"]
        avg_valence = sum(e['valence'] for e in recent_experiences) / len(recent_experiences) if recent_experiences else 0
        self.traits['любопытство'] = min(1.0, max(0.0, 0.5 + avg_valence * 0.5))
        
        if new_lesson:
            self.lessons.append(new_lesson)
            if "ошиб" in new_lesson.lower() or "неудач" in new_lesson.lower():
                self.traits['смелость'] = max(0.0, self.traits['смелость'] - 0.1)

    def evaluate_action(self, action_desc: str, outcome_valence: float, model, tokenizer):
        """Метапознание: Оценка действия"""
        prompt = (
            f"Оцени это действие: {action_desc}. Исход (valence): {outcome_valence:.2f}. "
            "Было ли оно полезным/вредным? Кратко: 1-2 предложения на русском. "
            "Начни с 'Это действие было...'."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(inputs["input_ids"])
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )
        evaluation = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
        self.self_evaluations.append(evaluation)
        
        if "вредным" in evaluation.lower() or "ошиб" in evaluation.lower():
            self.traits['смелость'] -= 0.05
        elif "полезным" in evaluation.lower():
            self.traits['любопытство'] += 0.05
        
        return evaluation

@dataclass
class OtherModel:
    name: str = "Пользователь"
    traits: Dict[str, float] = None
    predicted_behavior: str = ""

    def __post_init__(self):
        if self.traits is None:
            self.traits = {'любопытство': 0.5, 'агрессивность': 0.0}

    def update_traits(self, user_text: str, model, tokenizer):
        """Обновление черт на основе текста пользователя"""
        prompt = (
            f"Оцени черты пользователя по тексту: '{user_text}'. "
            "Верни: любопытство: [0-1], агрессивность: [0-1]. Кратко."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(inputs["input_ids"])
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=50
        )
        eval_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
        
        try:
            parts = eval_text.split(',')
            self.traits['любопытство'] = float(parts[0].split(':')[1].strip())
            self.traits['агрессивность'] = float(parts[1].split(':')[1].strip())
        except:
            pass

    def predict_behavior(self, dialog_history: List[Dict], model, tokenizer):
        """Предсказание поведения пользователя"""
        history_summary = "\n".join(f"{msg['role']}: {msg['content']}" for msg in dialog_history[-3:])
        prompt = (
            f"На основе диалога: {history_summary}. "
            "Предскажи, что пользователь скажет или сделает дальше. Кратко: 1 предложение."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(inputs["input_ids"])
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=50
        )
        self.predicted_behavior = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
        return self.predicted_behavior

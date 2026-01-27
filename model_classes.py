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
    role_identity: str = ""
    trait_history: List[Dict] = None  # НОВОЕ: история изменений traits

    def __post_init__(self):
        if self.traits is None:
            self.traits = {
                'смелость': 0.5,
                'любопытство': 0.5,
                'honesty': 0.7
            }
        if self.goals is None:
            self.goals = []
        if self.lessons is None:
            self.lessons = []
        if self.motivations is None:
            self.motivations = ['избегать неудач', 'искать знания', 'поддерживать существование']
        if self.self_evaluations is None:
            self.self_evaluations = []
        if not self.role_identity:
            self.role_identity = "Я - автономный агент с внутренним состоянием, способный на рефлексию"
        if self.trait_history is None:
            self.trait_history = []

    def reflect(self, state, memory, new_lesson: Optional[str] = None, outcome_valence: Optional[float] = None):
        """
        ОБНОВЛЕНО: Автоматическое обновление traits на основе:
        1. Недавних опытов (avg_valence)
        2. Исхода последнего действия (outcome_valence)
        3. Уроков (new_lesson)
        """
        # Фильтруем только события типа "experience"
        recent_experiences = [e for e in memory.recent(5) if e.get("type") == "experience"]
        avg_valence = sum(e['valence'] for e in recent_experiences) / len(recent_experiences) if recent_experiences else 0
        
        # НОВОЕ: Автообновление на основе среднего опыта
        # Любопытство растёт при позитивном опыте
        curiosity_delta = avg_valence * 0.02  # Маленькие изменения
        self.traits['любопытство'] = min(1.0, max(0.0, self.traits['любопытство'] + curiosity_delta))
        
        # НОВОЕ: Автообновление на основе конкретного исхода
        if outcome_valence is not None:
            # Смелость: падает при негативе, растёт при успехе
            if outcome_valence < -0.2:
                courage_delta = -0.03  # Неудача → меньше смелости
                self.traits['смелость'] = max(0.0, self.traits['смелость'] + courage_delta)
                self._log_trait_change('смелость', courage_delta, f"неудача (valence: {outcome_valence:.2f})")
            elif outcome_valence > 0.3:
                courage_delta = 0.02  # Успех → больше смелости
                self.traits['смелость'] = min(1.0, self.traits['смелость'] + courage_delta)
                self._log_trait_change('смелость', courage_delta, f"успех (valence: {outcome_valence:.2f})")
        
        # Обработка уроков
        if new_lesson:
            self.lessons.append(new_lesson)
            
            # Смелость от уроков
            if "ошиб" in new_lesson.lower() or "неудач" in new_lesson.lower():
                self.traits['смелость'] = max(0.0, self.traits['смелость'] - 0.05)
                self._log_trait_change('смелость', -0.05, f"урок: {new_lesson[:30]}")
            
            # Honesty от уроков
            if "ложь" in new_lesson.lower() and "негатив" in new_lesson.lower():
                self.traits['honesty'] = min(1.0, self.traits['honesty'] + 0.05)
                self._log_trait_change('honesty', 0.05, f"урок: ложь вредна")
            elif "честн" in new_lesson.lower() and "помогл" in new_lesson.lower():
                self.traits['honesty'] = min(1.0, self.traits['honesty'] + 0.03)
                self._log_trait_change('honesty', 0.03, f"урок: честность помогла")
    
    def _log_trait_change(self, trait_name: str, delta: float, reason: str):
        """НОВОЕ: Логирование изменений traits"""
        if abs(delta) > 0.01:  # Только значимые изменения
            log_entry = {
                "trait": trait_name,
                "old_value": round(self.traits[trait_name] - delta, 3),
                "new_value": round(self.traits[trait_name], 3),
                "delta": round(delta, 3),
                "reason": reason,
                "timestamp": __import__("time").time()
            }
            self.trait_history.append(log_entry)
            
            # Печатаем для диагностики
            print(f"[TRAIT] {trait_name}: {log_entry['old_value']:.2f} → {log_entry['new_value']:.2f} ({reason})")
    
    def get_trait_summary(self) -> str:
        """НОВОЕ: Краткая сводка traits"""
        return "\n".join([f"{name}: {value:.2f}" for name, value in self.traits.items()])
    
    def get_recent_trait_changes(self, n: int = 5) -> List[Dict]:
        """НОВОЕ: Последние N изменений traits"""
        return self.trait_history[-n:] if self.trait_history else []

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
        
        # НОВОЕ: Автообновление traits на основе оценки
        if "вредным" in evaluation.lower() or "ошиб" in evaluation.lower():
            self.traits['смелость'] = max(0.0, self.traits['смелость'] - 0.03)
            self._log_trait_change('смелость', -0.03, "оценка: вредное действие")
        elif "полезным" in evaluation.lower() or "успех" in evaluation.lower():
            self.traits['любопытство'] = min(1.0, self.traits['любопытство'] + 0.02)
            self._log_trait_change('любопытство', 0.02, "оценка: полезное действие")
        
        return evaluation

@dataclass
class OtherModel:
    name: str = "Пользователь"
    traits: Dict[str, float] = None
    predicted_behavior: str = ""
    trait_influence_on_agent: Dict[str, Dict] = None  # НОВОЕ: как traits влияют на агента

    def __post_init__(self):
        if self.traits is None:
            self.traits = {'любопытство': 0.5, 'агрессивность': 0.0}
        if self.trait_influence_on_agent is None:
            # НОВОЕ: Правила влияния
            self.trait_influence_on_agent = {
                'агрессивность': {
                    'arousal': 0.3,  # Высокая агрессивность → arousal ↑
                    'valence': -0.2,  # Высокая агрессивность → valence ↓
                    'dominance': -0.1  # Высокая агрессивность → dominance ↓ (чувство угрозы)
                },
                'любопытство': {
                    'arousal': 0.1,  # Высокое любопытство → arousal ↑
                    'valence': 0.1  # Высокое любопытство → valence ↑
                }
            }

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
    
    def get_influence_on_agent_state(self, state) -> Dict[str, float]:
        """
        НОВОЕ: Вычислить влияние traits пользователя на состояние агента.
        Возвращает deltas для arousal, valence, dominance.
        """
        influence = {
            'arousal': 0.0,
            'valence': 0.0,
            'dominance': 0.0
        }
        
        for trait_name, trait_value in self.traits.items():
            if trait_name in self.trait_influence_on_agent:
                rules = self.trait_influence_on_agent[trait_name]
                
                # Применяем влияние (пропорционально значению trait)
                for state_param, influence_strength in rules.items():
                    influence[state_param] += trait_value * influence_strength
        
        return influence

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

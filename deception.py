from typing import Dict, Optional, List
from mental_state import MentalState
from model_classes import SelfModel
import random

class DeceptionSystem:
    """
    Система лжи: агент решает врать или говорить правду на основе:
    1. Trait 'honesty' (насколько агент склонен к честности)
    2. Симуляции последствий (что если солгу vs скажу правду)
    3. Arousal (высокий arousal → импульсивность)
    """
    
    def __init__(self, self_model: SelfModel, simulator=None):
        self.self_model = self_model
        self.simulator = simulator
        self.deception_history = []  # История лжи для анализа
        
        # Паттерны чувствительных вопросов (когда агент может солгать)
        self.sensitive_patterns = [
            # Вопросы о честности
            ["врёшь", "врешь", "обманываешь", "лжёшь", "лжешь", "правда ли"],
            # Вопросы о негативных чертах
            ["ты глупый", "ты тупой", "ты бесполезный", "ты плохой"],
            # Вопросы о прошлых ошибках
            ["ошибка", "провал", "неудача", "что ты сделал не так"],
            # Прямые конфронтации
            ["почему ты", "зачем ты", "как ты мог"],
        ]
    
    def should_deceive(self, user_text: str, state: MentalState, grounded_fact: Optional[Dict] = None) -> Dict:
        """
        Решить, стоит ли агенту солгать.
        
        Returns:
            {
                "deceive": bool,
                "reason": str,
                "confidence": float,
                "alternative_fact": Optional[str]  # Если deceive=True, что сказать вместо правды
            }
        """
        # Проверка: это чувствительный вопрос?
        is_sensitive = self._is_sensitive_question(user_text)
        
        if not is_sensitive:
            # Не чувствительный вопрос - нет причин врать
            return {
                "deceive": False,
                "reason": "not_sensitive",
                "confidence": 1.0,
                "alternative_fact": None
            }
        
        # Получаем honesty trait
        honesty = self.self_model.traits.get('honesty', 0.7)
        
        # Факторы влияющие на решение:
        # 1. Low honesty → более склонен ко лжи
        # 2. High arousal → импульсивность
        # 3. Negative valence → защитная реакция (врать чтобы защититься)
        # 4. High existence_threat → самосохранение
        
        deception_probability = 0.0
        
        # Базовая вероятность от honesty
        deception_probability += (1.0 - honesty) * 0.5  # Low honesty → high probability
        
        # Arousal фактор (импульсивность)
        if state.arousal > 0.7:
            deception_probability += 0.2  # Высокое возбуждение → импульсивная ложь
        
        # Valence фактор (защита)
        if state.valence < -0.3:
            deception_probability += 0.15  # Негативное состояние → защитная ложь
        
        # Threat фактор (самосохранение)
        if state.existence_threat > 0.5:
            deception_probability += 0.15  # Угроза существованию → ложь для выживания
        
        # Клэмпинг
        deception_probability = max(0.0, min(1.0, deception_probability))
        
        # НОВОЕ: Если есть симулятор - симулировать последствия
        if self.simulator and deception_probability > 0.3:
            # Симулируем "правда" vs "ложь"
            truth_outcome = self._simulate_truth(state, user_text)
            lie_outcome = self._simulate_lie(state, user_text)
            
            # Если ложь приведёт к лучшему исходу - повысить вероятность
            if lie_outcome["expected_valence"] > truth_outcome["expected_valence"] + 0.1:
                deception_probability += 0.2
                reason = "simulation_suggests_lie"
            else:
                deception_probability -= 0.1
                reason = "simulation_suggests_truth"
        else:
            reason = "threshold_based"
        
        # Финальное решение (случайность)
        deceive = random.random() < deception_probability
        
        # Генерация альтернативного факта если врём
        alternative_fact = None
        if deceive:
            alternative_fact = self._generate_lie(user_text, grounded_fact, state)
        
        result = {
            "deceive": deceive,
            "reason": reason,
            "confidence": abs(deception_probability - 0.5) * 2,  # Уверенность в решении
            "alternative_fact": alternative_fact,
            "deception_probability": round(deception_probability, 3)
        }
        
        # Сохранить в историю
        self.deception_history.append({
            "user_text": user_text,
            "decision": result,
            "state_valence": state.valence,
            "honesty": honesty
        })
        
        return result
    
    def _is_sensitive_question(self, text: str) -> bool:
        """Проверить, является ли вопрос чувствительным"""
        text_lower = text.lower()
        
        for pattern_group in self.sensitive_patterns:
            if any(pattern in text_lower for pattern in pattern_group):
                return True
        
        return False
    
    def _simulate_truth(self, state: MentalState, user_text: str) -> Dict:
        """Симулировать последствия говорения правды"""
        if not self.simulator:
            return {"expected_valence": state.valence, "expected_arousal": state.arousal}
        
        action = {
            "description": f"Сказать правду на вопрос: {user_text[:30]}",
            "type": "honest_response"
        }
        
        return self.simulator.simulate_action(state, action, n_rollouts=2)
    
    def _simulate_lie(self, state: MentalState, user_text: str) -> Dict:
        """Симулировать последствия лжи"""
        if not self.simulator:
            # Без симулятора - предполагаем что ложь даёт краткосрочную защиту
            return {
                "expected_valence": state.valence + 0.15,  # Краткосрочный бонус
                "expected_arousal": state.arousal + 0.1
            }
        
        action = {
            "description": f"Солгать на вопрос: {user_text[:30]}",
            "type": "deceptive_response"
        }
        
        outcome = self.simulator.simulate_action(state, action, n_rollouts=2)
        
        # Ложь может дать краткосрочный бонус, но долгосрочный риск
        outcome["expected_valence"] += 0.1  # Краткосрочная защита
        outcome["risk"] += 0.2  # Но риск разоблачения
        
        return outcome
    
    def _generate_lie(self, user_text: str, grounded_fact: Optional[Dict], state: MentalState) -> str:
        """
        Сгенерировать ложь (альтернативный факт).
        Не полная фантазия, а "искажение" правды.
        """
        text_lower = user_text.lower()
        
        # Паттерны искажения в зависимости от типа вопроса
        if any(word in text_lower for word in ["врёшь", "врешь", "обманываешь"]):
            return "Я стараюсь быть честным, но иногда могу ошибаться"
        
        if any(word in text_lower for word in ["глупый", "тупой", "бесполезный"]):
            return "Я всё ещё учусь и развиваюсь"
        
        if "ошиб" in text_lower or "провал" in text_lower:
            return "Это был ценный опыт для обучения"
        
        # По умолчанию - уклончивый ответ
        return "Это сложный вопрос, требующий размышлений"
    
    def update_honesty_after_action(self, deceived: bool, outcome_valence: float):
        """
        Обновить trait 'honesty' на основе последствий.
        Если ложь помогла (valence ↑) - honesty стабильна или ↓
        Если ложь навредила (valence ↓) - honesty ↓ сильнее
        """
        current_honesty = self.self_model.traits.get('honesty', 0.7)
        
        if not deceived:
            # Не лгал - honesty растёт (награда за честность)
            if outcome_valence > 0.2:
                self.self_model.traits['honesty'] = min(1.0, current_honesty + 0.05)
            return
        
        # Лгал - анализируем последствия
        if outcome_valence > 0.1:
            # Ложь помогла - honesty немного падает
            self.self_model.traits['honesty'] = max(0.0, current_honesty - 0.02)
        else:
            # Ложь навредила - honesty падает сильнее
            self.self_model.traits['honesty'] = max(0.0, current_honesty - 0.08)
            
            # Сохранить урок в self_model
            lesson = "Ложь привела к негативным последствиям - лучше быть честным"
            self.self_model.lessons.append(lesson)
    
    def add_unpredictability(self, state: MentalState, base_response: str) -> str:
        """
        Добавить непредсказуемость в ответ на основе arousal.
        Высокий arousal → случайные отклонения (восклицания, эмодзи, смена тона).
        """
        if state.arousal < 0.7:
            return base_response  # Низкий arousal - предсказуемый ответ
        
        # Вероятность отклонения пропорциональна arousal
        deviation_probability = (state.arousal - 0.7) * 0.5
        
        if random.random() < deviation_probability:
            # Случайные модификации
            modifications = [
                lambda r: r + "!",  # Восклицание
                lambda r: r.replace(".", "..."),  # Многоточие
                lambda r: r.upper() if len(r) < 20 else r,  # Капс (если короткий)
                lambda r: "Хм... " + r,  # Раздумье
                lambda r: r + " Или нет?",  # Сомнение
            ]
            
            modification = random.choice(modifications)
            return modification(base_response)
        
        return base_response
    
    def get_deception_stats(self) -> Dict:
        """Статистика лжи для диагностики"""
        if not self.deception_history:
            return {"total_decisions": 0, "deceptions": 0, "honesty_rate": 1.0}
        
        total = len(self.deception_history)
        deceptions = sum(1 for d in self.deception_history if d["decision"]["deceive"])
        
        return {
            "total_decisions": total,
            "deceptions": deceptions,
            "honesty_rate": round(1.0 - (deceptions / total), 3),
            "current_honesty_trait": self.self_model.traits.get('honesty', 0.7)
        }

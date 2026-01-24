from typing import Dict, List, Optional
from mental_state import MentalState
from model_classes import SelfModel
from systems import FutureExpectationSystem
import random

class WorldSimulator:
    """
    Симуляция мира: "что если" сценарии для планирования.
    Агент проигрывает действия в голове, предсказывая последствия.
    """
    
    def __init__(self, future_system: FutureExpectationSystem, self_model: SelfModel):
        self.future_system = future_system
        self.self_model = self_model
        self.simulation_cache = {}  # Кэш симуляций для оптимизации
        self.simulation_count = 0
    
    def simulate_action(self, current_state: MentalState, action: Dict, n_rollouts: int = 3) -> Dict:
        """
        Симулировать действие N раз методом Монте-Карло.
        
        Args:
            current_state: Текущее состояние агента
            action: Действие для симуляции {"description": str, "type": str}
            n_rollouts: Количество симуляций (по умолчанию 3)
        
        Returns:
            {
                "expected_valence": float,
                "expected_arousal": float,
                "expected_threat": float,
                "confidence": float,  # Уверенность в предсказании
                "risk": float  # Риск негативного исхода
            }
        """
        # Проверка кэша
        cache_key = f"{action['description']}_{current_state.valence:.2f}_{current_state.arousal:.2f}"
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]
        
        outcomes = []
        
        for i in range(n_rollouts):
            outcome = self._rollout(current_state, action, rollout_id=i)
            outcomes.append(outcome)
        
        # Усреднение результатов
        avg_valence = sum(o["valence"] for o in outcomes) / len(outcomes)
        avg_arousal = sum(o["arousal"] for o in outcomes) / len(outcomes)
        avg_threat = sum(o["threat"] for o in outcomes) / len(outcomes)
        
        # Вычисление уверенности (low variance = high confidence)
        variance_valence = sum((o["valence"] - avg_valence)**2 for o in outcomes) / len(outcomes)
        confidence = max(0.0, 1.0 - variance_valence)  # Низкая дисперсия = высокая уверенность
        
        # Вычисление риска (вероятность негативного исхода)
        negative_outcomes = sum(1 for o in outcomes if o["valence"] < current_state.valence)
        risk = negative_outcomes / len(outcomes)
        
        result = {
            "expected_valence": round(avg_valence, 3),
            "expected_arousal": round(avg_arousal, 3),
            "expected_threat": round(avg_threat, 3),
            "confidence": round(confidence, 3),
            "risk": round(risk, 3)
        }
        
        # Сохранение в кэш
        self.simulation_cache[cache_key] = result
        self.simulation_count += n_rollouts
        
        return result
    
    def _rollout(self, state: MentalState, action: Dict, rollout_id: int = 0) -> Dict:
        """
        Один rollout: симуляция одного исхода действия.
        
        Использует future_system для предсказания реакции,
        добавляет случайность для моделирования неопределённости.
        """
        action_desc = action.get("description", "неизвестное действие")
        
        # Предсказание исхода через future_system
        prediction = self.future_system.predict(action_desc)
        
        # Добавление случайности (stochastic world)
        noise_valence = random.uniform(-0.1, 0.1)
        noise_arousal = random.uniform(-0.05, 0.05)
        
        # Вычисление нового состояния
        new_valence = state.valence + prediction['valence'] + noise_valence
        new_arousal = state.arousal + prediction['arousal'] + noise_arousal
        
        # Клэмпинг в допустимые границы
        new_valence = max(-1.0, min(1.0, new_valence))
        new_arousal = max(0.0, min(1.0, new_arousal))
        
        # Предсказание threat на основе valence и типа действия
        threat_delta = 0.0
        if new_valence < -0.3:
            threat_delta = 0.1  # Негативная valence повышает threat
        if "избегать" in action_desc.lower() or "защит" in action_desc.lower():
            threat_delta -= 0.05  # Защитные действия понижают threat
        
        new_threat = max(0.0, min(1.0, state.existence_threat + threat_delta))
        
        return {
            "valence": new_valence,
            "arousal": new_arousal,
            "threat": new_threat,
            "action": action_desc
        }
    
    def compare_actions(self, current_state: MentalState, actions: List[Dict]) -> List[Dict]:
        """
        Сравнить несколько действий и ранжировать их по utility.
        
        Returns:
            Список действий с добавленными полями "simulation" и "utility"
        """
        if not actions:
            return []
        
        results = []
        
        for action in actions:
            simulation = self.simulate_action(current_state, action)
            
            # Вычисление utility (полезности)
            # Формула: valence важнее всего, низкий arousal и threat - бонус
            utility = (
                simulation["expected_valence"] * 0.6 +  # Главный фактор
                (1.0 - simulation["expected_arousal"]) * 0.2 +  # Спокойствие - хорошо
                (1.0 - simulation["expected_threat"]) * 0.2  # Безопасность - хорошо
            )
            
            # Штраф за риск
            utility -= simulation["risk"] * 0.3
            
            # Бонус за уверенность
            utility += simulation["confidence"] * 0.1
            
            results.append({
                "action": action,
                "simulation": simulation,
                "utility": round(utility, 3)
            })
        
        # Сортировка по убыванию utility
        results.sort(key=lambda x: x["utility"], reverse=True)
        
        return results
    
    def get_simulation_stats(self) -> Dict:
        """
        Статистика симуляций для диагностики.
        """
        return {
            "total_simulations": self.simulation_count,
            "cache_size": len(self.simulation_cache)
        }
    
    def clear_cache(self):
        """
        Очистить кэш симуляций (использовать если мир сильно изменился).
        """
        self.simulation_cache.clear()
        print(f"[СИМУЛЯТОР] Кэш очищен. Было симуляций: {self.simulation_count}")

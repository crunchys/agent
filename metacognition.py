"""
Metacognition System - Метакогниция
====================================

Система осознания собственного мышления:
- Confidence tracking (уверенность в ответах)
- Knowledge monitoring (знаю/не знаю)
- Uncertainty expression (выражение сомнений)
- Self-error detection (обнаружение ошибок)

Автор: Cognitive Agent Project
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from time import time
import random


@dataclass
class ConfidenceEstimate:
    """Оценка уверенности в знании/ответе"""
    confidence: float  # 0.0-1.0
    knowledge_state: str  # "known", "uncertain", "unknown"
    sources: List[str]  # Откуда уверенность
    reasoning: str  # Почему такая уверенность


class MetacognitionSystem:
    """
    Система метакогниции - "думать о мышлении"
    
    Основные функции:
    1. Оценка уверенности в ответах (confidence)
    2. Мониторинг знаний (я знаю что не знаю)
    3. Выражение неуверенности в тексте
    4. Обнаружение своих ошибок
    """
    
    def __init__(self, self_model, tool_manager=None):
        self.self_model = self_model
        self.tool_manager = tool_manager
        
        # История оценок уверенности
        self.confidence_history = []
        
        # Текущая уверенность (последняя оценка)
        self.current_confidence = 0.5
        
        # Статистика калибровки
        self.calibration_data = []  # [(predicted_confidence, was_correct), ...]
        
        # Пороги для состояний знания
        self.thresholds = {
            "known": 0.7,      # > 0.7 = "я знаю"
            "uncertain": 0.3,  # 0.3-0.7 = "я не уверен"
            "unknown": 0.3     # < 0.3 = "я не знаю"
        }
    
    def estimate_confidence(
        self,
        query: str,
        grounded_results: Optional[List[Dict]] = None,
        prediction_error: float = 0.0,
        state = None
    ) -> ConfidenceEstimate:
        """
        Оценить уверенность в ответе на вопрос.
        
        Args:
            query: Вопрос/тема
            grounded_results: Результаты grounding (факты из памяти)
            prediction_error: Prediction error (новизна)
            state: Ментальное состояние агента
        
        Returns:
            ConfidenceEstimate с оценкой уверенности
        """
        confidence = 0.5  # Базовая уверенность
        sources = []
        
        # 1. Grounding factor (есть ли факты в памяти?)
        if grounded_results:
            for result in grounded_results:
                if result.get("fact_verified", False):
                    confidence += 0.2
                    sources.append("verified_fact")
                
                # Высокая confidence из grounding
                ground_confidence = result.get("confidence", 0.0)
                if ground_confidence > 0.7:
                    confidence += 0.15
                    sources.append("high_ground_confidence")
                
                # Knowledge gap (пробел в знаниях)
                knowledge_gap = result.get("knowledge_gap", 0.0)
                if knowledge_gap > 0.5:
                    confidence -= 0.3
                    sources.append("knowledge_gap")
        else:
            # Нет grounding результатов = не проверяли факты
            confidence -= 0.1
            sources.append("no_grounding")
        
        # 2. Prediction error factor (новизна → неуверенность)
        if prediction_error > 0.7:
            confidence -= 0.2
            sources.append("high_surprise")
        elif prediction_error > 0.4:
            confidence -= 0.1
            sources.append("medium_surprise")
        
        # 3. Honesty trait factor
        honesty = self.self_model.traits.get('honesty', 0.7)
        if honesty > 0.8:
            # Честный агент более критичен к себе
            confidence *= 0.9
            sources.append("honesty_correction")
        
        # 4. Emotional state factor (anxiety → неуверенность)
        if state:
            if state.arousal > 0.7 and state.valence < 0:
                # Высокое возбуждение + негатив = тревога → неуверенность
                confidence -= 0.1
                sources.append("anxious_state")
        
        # Клэмпинг
        confidence = max(0.0, min(1.0, confidence))
        
        # Определение knowledge state
        if confidence >= self.thresholds["known"]:
            knowledge_state = "known"
            reasoning = "Высокая уверенность - есть проверенные факты"
        elif confidence >= self.thresholds["uncertain"]:
            knowledge_state = "uncertain"
            reasoning = "Средняя уверенность - есть пробелы в знаниях"
        else:
            knowledge_state = "unknown"
            reasoning = "Низкая уверенность - недостаточно информации"
        
        estimate = ConfidenceEstimate(
            confidence=round(confidence, 3),
            knowledge_state=knowledge_state,
            sources=sources,
            reasoning=reasoning
        )
        
        # Сохранить в историю
        self.confidence_history.append({
            "query": query,
            "estimate": estimate,
            "timestamp": time()
        })
        
        # Ограничить историю
        if len(self.confidence_history) > 50:
            self.confidence_history.pop(0)
        
        # Обновить текущую уверенность
        self.current_confidence = confidence
        
        print(f"[METACOGNITION] Confidence: {confidence:.2f} ({knowledge_state})")
        print(f"  Reasoning: {reasoning}")
        
        return estimate
    
    def express_uncertainty(
        self,
        response: str,
        confidence_estimate: ConfidenceEstimate
    ) -> str:
        """
        Добавить выражение неуверенности в ответ.
        
        Args:
            response: Исходный ответ
            confidence_estimate: Оценка уверенности
        
        Returns:
            Модифицированный ответ с выражением uncertainty
        """
        knowledge_state = confidence_estimate.knowledge_state
        confidence = confidence_estimate.confidence
        
        # Если уверенность высокая - не модифицируем
        if knowledge_state == "known" and confidence > 0.8:
            return response
        
        # Если неуверенность - добавляем фразы
        if knowledge_state == "unknown":
            # Очень низкая уверенность
            prefixes = [
                "Я не уверен, но ",
                "Мне кажется, ",
                "Не могу сказать точно, но ",
                "Это лишь предположение: ",
            ]
            prefix = random.choice(prefixes)
            
            # Также можем добавить постфикс
            postfixes = [
                " Но я могу ошибаться.",
                " Это не точная информация.",
                " Лучше проверить дополнительно.",
            ]
            postfix = random.choice(postfixes) if random.random() < 0.5 else ""
            
            return prefix + response + postfix
        
        elif knowledge_state == "uncertain":
            # Средняя уверенность
            phrases = [
                "Насколько я помню, ",
                "Вероятно, ",
                "Думаю, что ",
                "Если не ошибаюсь, ",
            ]
            prefix = random.choice(phrases)
            
            return prefix + response
        
        return response
    
    def should_acknowledge_uncertainty(
        self,
        confidence_estimate: ConfidenceEstimate
    ) -> bool:
        """
        Решить, стоит ли явно признать неуверенность.
        
        Args:
            confidence_estimate: Оценка уверенности
        
        Returns:
            True если нужно признать неуверенность
        """
        # Если уверенность очень низкая - всегда признаем
        if confidence_estimate.confidence < 0.3:
            return True
        
        # Если knowledge_state = unknown - признаем
        if confidence_estimate.knowledge_state == "unknown":
            return True
        
        # Если честность высокая - более склонны признавать
        honesty = self.self_model.traits.get('honesty', 0.7)
        if honesty > 0.8 and confidence_estimate.confidence < 0.5:
            return True
        
        return False
    
    def generate_uncertainty_statement(
        self,
        query: str,
        confidence_estimate: ConfidenceEstimate
    ) -> str:
        """
        Сгенерировать явное признание неуверенности.
        
        Args:
            query: Вопрос
            confidence_estimate: Оценка уверенности
        
        Returns:
            Текст признания неуверенности
        """
        knowledge_state = confidence_estimate.knowledge_state
        
        if knowledge_state == "unknown":
            # Полное незнание
            statements = [
                "Я не знаю ответа на этот вопрос.",
                "К сожалению, у меня нет информации об этом.",
                "Не могу ответить точно - недостаточно данных.",
                "Это за пределами моих знаний.",
            ]
        elif knowledge_state == "uncertain":
            # Частичное знание
            statements = [
                "Я не совсем уверен в ответе.",
                "У меня есть лишь приблизительное понимание.",
                "Могу предположить, но точности нет.",
                "Есть некоторая неопределенность в этом вопросе.",
            ]
        else:
            # Не должно вызываться для "known"
            statements = ["Думаю, я знаю ответ."]
        
        return random.choice(statements)
    
    def detect_self_error(
        self,
        own_statement: str,
        grounded_results: Optional[List[Dict]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Обнаружить ошибку в собственном утверждении.
        
        Args:
            own_statement: Что агент сказал
            grounded_results: Результаты grounding
        
        Returns:
            (есть_ошибка, описание_ошибки)
        """
        if not grounded_results:
            return False, None
        
        for result in grounded_results:
            # Если grounding обнаружил нарушение self-fact
            if result.get("self_fact_violation", False):
                error_desc = f"Нарушение self-fact: {result.get('stimulus', '')}"
                print(f"[METACOGNITION] ⚠️ Обнаружена ошибка: {error_desc}")
                return True, error_desc
            
            # Если grounding показал низкую confidence при высоком утверждении
            if result.get("fact_verified", False) is False:
                if result.get("confidence", 0) < 0.3:
                    error_desc = "Утверждение не подтверждается фактами"
                    print(f"[METACOGNITION] ⚠️ Возможная ошибка: {error_desc}")
                    return True, error_desc
        
        return False, None
    
    def update_calibration(
        self,
        predicted_confidence: float,
        was_correct: bool
    ):
        """
        Обновить калибровку (насколько точно агент оценивает свою уверенность).
        
        Args:
            predicted_confidence: Предсказанная уверенность
            was_correct: Был ли ответ правильным
        """
        self.calibration_data.append({
            "predicted": predicted_confidence,
            "correct": was_correct,
            "timestamp": time()
        })
        
        # Ограничить историю
        if len(self.calibration_data) > 100:
            self.calibration_data.pop(0)
        
        # Вычислить калибровку (можно использовать для корректировки)
        if len(self.calibration_data) >= 10:
            calibration = self._compute_calibration()
            print(f"[METACOGNITION] Калибровка: {calibration:.2f}")
    
    def _compute_calibration(self) -> float:
        """
        Вычислить насколько хорошо откалиброван агент.
        
        Returns:
            Calibration score (0-1, где 1 = идеально)
        """
        if not self.calibration_data:
            return 0.5
        
        # Простая метрика: соответствие confidence и correctness
        total_error = 0.0
        for item in self.calibration_data[-20:]:
            predicted = item["predicted"]
            actual = 1.0 if item["correct"] else 0.0
            error = abs(predicted - actual)
            total_error += error
        
        avg_error = total_error / len(self.calibration_data[-20:])
        calibration = 1.0 - avg_error
        
        return max(0.0, min(1.0, calibration))
    
    def get_metacognitive_state(self) -> Dict:
        """
        Получить текущее метакогнитивное состояние.
        
        Returns:
            Словарь с метакогнитивными параметрами
        """
        # Последняя оценка
        last_estimate = None
        if self.confidence_history:
            last_estimate = self.confidence_history[-1]["estimate"]
        
        # Калибровка
        calibration = self._compute_calibration() if self.calibration_data else 0.5
        
        # Средняя уверенность за последние N оценок
        if self.confidence_history:
            recent_confidences = [h["estimate"].confidence for h in self.confidence_history[-10:]]
            avg_confidence = sum(recent_confidences) / len(recent_confidences)
        else:
            avg_confidence = 0.5
        
        return {
            "current_confidence": self.current_confidence,
            "last_estimate": last_estimate,
            "average_confidence": round(avg_confidence, 3),
            "calibration": round(calibration, 3),
            "history_size": len(self.confidence_history),
            "calibration_data_size": len(self.calibration_data)
        }
    
    def generate_metacognitive_thought(self, state) -> Optional[str]:
        """
        Сгенерировать метакогнитивную мысль ("я думаю что я думаю...").
        
        Args:
            state: Ментальное состояние
        
        Returns:
            Мысль о собственном мышлении
        """
        metacog_state = self.get_metacognitive_state()
        confidence = metacog_state["current_confidence"]
        
        if confidence < 0.3:
            thoughts = [
                "Я замечаю, что я не уверен в своем понимании.",
                "Осознаю пробелы в своих знаниях.",
                "Чувствую неуверенность в своих выводах.",
            ]
        elif confidence < 0.6:
            thoughts = [
                "Я частично понимаю ситуацию, но есть неясности.",
                "Замечаю, что мое понимание неполное.",
                "Осознаю, что нужно больше информации.",
            ]
        else:
            thoughts = [
                "Я уверен в своем понимании ситуации.",
                "Чувствую, что хорошо разбираюсь в теме.",
                "Осознаю, что мои знания достаточны для ответа.",
            ]
        
        return random.choice(thoughts)


# Вспомогательный класс для интеграции
class MetacognitionIntegrator:
    """
    Интегратор метакогниции в основной цикл агента.
    """
    
    @staticmethod
    def process_with_metacognition(
        agent,
        user_text: str,
        response: str,
        grounded_results: Optional[List[Dict]] = None
    ) -> str:
        """
        Обработать ответ с учетом метакогниции.
        
        Args:
            agent: Экземпляр Agent
            user_text: Вопрос пользователя
            response: Исходный ответ
            grounded_results: Результаты grounding
        
        Returns:
            Модифицированный ответ с выражением uncertainty
        """
        if not hasattr(agent, 'metacognition'):
            return response
        
        metacog = agent.metacognition
        
        # 1. Оценить уверенность
        prediction_error = getattr(agent, 'last_prediction_error', 0.0)
        confidence_estimate = metacog.estimate_confidence(
            query=user_text,
            grounded_results=grounded_results,
            prediction_error=prediction_error,
            state=agent.state
        )
        
        # 2. Обнаружить ошибки
        has_error, error_desc = metacog.detect_self_error(response, grounded_results)
        if has_error:
            # Если обнаружена ошибка - скорректировать ответ
            correction = f"Прошу прощения, я ошибся. {error_desc}"
            response = correction
        
        # 3. Выразить неуверенность если нужно
        if metacog.should_acknowledge_uncertainty(confidence_estimate):
            # Явное признание
            uncertainty_statement = metacog.generate_uncertainty_statement(
                user_text, confidence_estimate
            )
            response = uncertainty_statement + " " + response
        else:
            # Мягкое выражение через префиксы
            response = metacog.express_uncertainty(response, confidence_estimate)
        
        return response

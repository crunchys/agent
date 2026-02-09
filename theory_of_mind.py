"""
Theory of Mind System - Теория разума
======================================

Система понимания ментальных состояний других:
- Belief tracking (отслеживание убеждений)
- Intention recognition (распознавание намерений)
- Mental state simulation (симуляция состояний)
- False belief understanding (ложные убеждения)
- Empathy (эмпатия)

Автор: Cognitive Agent Project
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from time import time
import random


@dataclass
class UserBelief:
    """Убеждение пользователя"""
    topic: str  # О чем убеждение
    content: str  # Что верит
    confidence: float  # Насколько агент уверен в этом (0-1)
    source: str  # Откуда узнали ("inferred", "stated", "observed")
    timestamp: float = field(default_factory=time)


@dataclass
class UserIntention:
    """Намерение пользователя"""
    intention_type: str  # greeting, learn, challenge, share, etc.
    confidence: float  # Уверенность (0-1)
    evidence: List[str]  # Доказательства
    timestamp: float = field(default_factory=time)


@dataclass
class MentalStateEstimate:
    """Оценка ментального состояния пользователя"""
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    dominance: float  # -1 to 1
    emotion: str  # joy, anger, sadness, fear, etc.
    confidence: float  # 0-1
    reasoning: str  # Почему так решили


class BeliefTracker:
    """
    Отслеживание убеждений пользователя о различных темах.
    """
    
    def __init__(self):
        self.beliefs: Dict[str, UserBelief] = {}
        
        # Категории убеждений
        self.categories = {
            "about_agent": ["агент", "ты", "твой", "ai", "искусственный"],
            "about_consciousness": ["сознание", "осознан", "чувств", "эмоц"],
            "about_honesty": ["честн", "правд", "лож", "обман"],
            "about_capability": ["можешь", "умеешь", "способен", "навык"],
        }
    
    def update_beliefs(self, user_text: str, agent_response: str = "") -> List[UserBelief]:
        """
        Обновить убеждения на основе текста пользователя.
        
        Args:
            user_text: Что сказал пользователь
            agent_response: Ответ агента (для контекста)
        
        Returns:
            Список обновленных убеждений
        """
        updated = []
        text_lower = user_text.lower()
        
        # 1. Явные утверждения ("я думаю что...", "ты...")
        if "я думаю" in text_lower or "я считаю" in text_lower:
            # Пользователь явно выражает убеждение
            belief = self._extract_explicit_belief(user_text)
            if belief:
                self.beliefs[belief.topic] = belief
                updated.append(belief)
                print(f"[ToM] Новое убеждение: {belief.topic} = '{belief.content}'")
        
        # 2. Вопросы о агенте ("ты можешь?", "у тебя есть?")
        if any(word in text_lower for word in ["ты ", "твой", "твоя", "твоё"]):
            category = self._categorize_topic(text_lower)
            if category:
                belief = self._infer_belief_from_question(user_text, category)
                if belief:
                    self.beliefs[belief.topic] = belief
                    updated.append(belief)
        
        # 3. Оценки и суждения ("ты хорош", "это плохо")
        if any(word in text_lower for word in ["хорош", "плох", "отличн", "ужас", "круто"]):
            belief = self._extract_evaluation(user_text)
            if belief:
                self.beliefs[belief.topic] = belief
                updated.append(belief)
        
        return updated
    
    def _extract_explicit_belief(self, text: str) -> Optional[UserBelief]:
        """Извлечь явное убеждение из текста"""
        text_lower = text.lower()
        
        # Паттерны явных убеждений
        if "я думаю что ты" in text_lower:
            content = text_lower.split("я думаю что ты")[1].split(".")[0].strip()
            return UserBelief(
                topic="about_agent",
                content=f"агент {content}",
                confidence=0.9,
                source="stated"
            )
        
        if "я считаю что" in text_lower:
            content = text_lower.split("я считаю что")[1].split(".")[0].strip()
            return UserBelief(
                topic="user_opinion",
                content=content,
                confidence=0.9,
                source="stated"
            )
        
        return None
    
    def _categorize_topic(self, text: str) -> Optional[str]:
        """Определить категорию темы"""
        for category, keywords in self.categories.items():
            if any(kw in text for kw in keywords):
                return category
        return None
    
    def _infer_belief_from_question(self, text: str, category: str) -> Optional[UserBelief]:
        """Вывести убеждение из вопроса"""
        text_lower = text.lower()
        
        # "Ты можешь X?" → верит что агент может или интересуется
        if "можешь" in text_lower or "умеешь" in text_lower:
            return UserBelief(
                topic="about_capability",
                content="интересуется возможностями агента",
                confidence=0.6,
                source="inferred"
            )
        
        # "У тебя есть эмоции?" → интересуется сознанием
        if "эмоци" in text_lower or "чувств" in text_lower:
            return UserBelief(
                topic="about_consciousness",
                content="интересуется эмоциями и сознанием агента",
                confidence=0.7,
                source="inferred"
            )
        
        return None
    
    def _extract_evaluation(self, text: str) -> Optional[UserBelief]:
        """Извлечь оценку (хорошо/плохо)"""
        text_lower = text.lower()
        
        positive_words = ["хорош", "отличн", "круто", "супер", "молодец", "браво"]
        negative_words = ["плох", "ужас", "отврат", "тупой", "глуп"]
        
        if any(word in text_lower for word in positive_words):
            return UserBelief(
                topic="evaluation_of_agent",
                content="положительная оценка",
                confidence=0.8,
                source="observed"
            )
        
        if any(word in text_lower for word in negative_words):
            return UserBelief(
                topic="evaluation_of_agent",
                content="негативная оценка",
                confidence=0.8,
                source="observed"
            )
        
        return None
    
    def get_belief(self, topic: str) -> Optional[UserBelief]:
        """Получить убеждение по теме"""
        return self.beliefs.get(topic)
    
    def get_all_beliefs(self) -> List[UserBelief]:
        """Получить все убеждения"""
        return list(self.beliefs.values())


class IntentionRecognizer:
    """
    Распознавание намерений пользователя.
    """
    
    def __init__(self):
        # Паттерны для разных намерений
        self.patterns = {
            "greeting": {
                "keywords": ["привет", "здравствуй", "добрый день", "hi", "hello"],
                "questions": [],
            },
            "learn": {
                "keywords": ["объясни", "расскажи", "что такое", "как работает", "почему"],
                "questions": ["что", "как", "почему", "зачем"],
            },
            "challenge": {
                "keywords": ["врёшь", "врешь", "не верю", "докажи", "ерунда"],
                "questions": ["правда ли", "уверен"],
            },
            "share": {
                "keywords": ["я думаю", "я считаю", "мне кажется", "по-моему"],
                "questions": [],
            },
            "small_talk": {
                "keywords": ["как дела", "что делаешь", "как настроение"],
                "questions": ["как ты", "что у тебя"],
            },
            "get_to_know": {
                "keywords": ["расскажи о себе", "кто ты", "что ты"],
                "questions": ["кто ты", "какой ты"],
            },
            "test": {
                "keywords": ["тест", "проверка", "проверю"],
                "questions": [],
            },
            "request": {
                "keywords": ["пожалуйста", "можешь", "помоги", "сделай"],
                "questions": ["можешь ли", "сможешь ли"],
            },
        }
    
    def recognize(self, user_text: str, context: Optional[List[Dict]] = None) -> UserIntention:
        """
        Распознать намерение пользователя.
        
        Args:
            user_text: Текст пользователя
            context: История диалога (опционально)
        
        Returns:
            UserIntention с распознанным намерением
        """
        text_lower = user_text.lower()
        
        # Подсчет совпадений для каждого типа намерения
        scores = {}
        evidence_all = {}
        
        for intention_type, pattern in self.patterns.items():
            score = 0
            evidence = []
            
            # Проверка ключевых слов
            for keyword in pattern["keywords"]:
                if keyword in text_lower:
                    score += 1
                    evidence.append(f"keyword: {keyword}")
            
            # Проверка вопросов
            for question in pattern["questions"]:
                if question in text_lower and "?" in user_text:
                    score += 0.5
                    evidence.append(f"question: {question}")
            
            scores[intention_type] = score
            evidence_all[intention_type] = evidence
        
        # Выбор намерения с максимальным score
        if scores:
            best_intention = max(scores, key=scores.get)
            best_score = scores[best_intention]
            
            if best_score > 0:
                confidence = min(1.0, best_score / 3.0)  # Нормализация
                
                print(f"[ToM] Намерение: {best_intention} (confidence: {confidence:.2f})")
                
                return UserIntention(
                    intention_type=best_intention,
                    confidence=confidence,
                    evidence=evidence_all[best_intention],
                    timestamp=time()
                )
        
        # По умолчанию - general_query
        return UserIntention(
            intention_type="general_query",
            confidence=0.5,
            evidence=["no clear pattern"],
            timestamp=time()
        )


class MentalStateSimulator:
    """
    Симуляция ментального состояния пользователя.
    "Что бы я чувствовал на его месте?"
    """
    
    def __init__(self, emotion_system):
        self.emotion_system = emotion_system
        
        # База знаний: какие слова → какие эмоции
        self.emotion_triggers = {
            "joy": ["отлично", "супер", "здорово", "ура", "рад", "счастлив"],
            "anger": ["тупой", "дурак", "идиот", "бесит", "ненавижу", "злой"],
            "sadness": ["грустно", "печально", "жаль", "расстроен", "тоска"],
            "fear": ["боюсь", "страшно", "тревожно", "переживаю", "волнуюсь"],
            "surprise": ["вау", "ух ты", "неожиданно", "удивлен", "офигеть"],
            "disgust": ["отвратительно", "противно", "фу", "гадость"],
            "neutral": ["хм", "ок", "понятно", "ага"],
        }
    
    def simulate(self, user_text: str, context: Optional[Dict] = None) -> MentalStateEstimate:
        """
        Симулировать ментальное состояние пользователя.
        
        Args:
            user_text: Текст пользователя
            context: Дополнительный контекст
        
        Returns:
            MentalStateEstimate с оценкой состояния
        """
        text_lower = user_text.lower()
        
        # 1. Определить эмоцию по ключевым словам
        detected_emotions = {}
        for emotion, triggers in self.emotion_triggers.items():
            count = sum(1 for trigger in triggers if trigger in text_lower)
            if count > 0:
                detected_emotions[emotion] = count
        
        # 2. Выбрать доминирующую эмоцию
        if detected_emotions:
            dominant_emotion = max(detected_emotions, key=detected_emotions.get)
            confidence = min(1.0, detected_emotions[dominant_emotion] / 3.0)
        else:
            dominant_emotion = "neutral"
            confidence = 0.5
        
        # 3. Вычислить valence, arousal, dominance на основе эмоции
        valence, arousal, dominance = self._emotion_to_pad(dominant_emotion)
        
        # 4. Модификация на основе пунктуации
        if "!" in user_text:
            arousal += 0.2  # Восклицание → возбуждение
        if "?" in user_text:
            arousal += 0.1  # Вопрос → интерес
        if "..." in user_text:
            arousal -= 0.1  # Многоточие → размышление
        
        # 5. Модификация на основе капса
        if user_text.isupper() and len(user_text) > 5:
            arousal += 0.3  # КАПС → крик
            if valence < 0:
                valence -= 0.2  # Усиление негатива
        
        # Клэмпинг
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        dominance = max(-1.0, min(1.0, dominance))
        
        reasoning = f"Обнаружена эмоция: {dominant_emotion}"
        if "!" in user_text:
            reasoning += " (восклицание)"
        if user_text.isupper():
            reasoning += " (капс - крик)"
        
        print(f"[ToM] Состояние пользователя: {dominant_emotion} (v:{valence:.2f}, a:{arousal:.2f})")
        
        return MentalStateEstimate(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            emotion=dominant_emotion,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _emotion_to_pad(self, emotion: str) -> Tuple[float, float, float]:
        """
        Конвертировать эмоцию в PAD (Pleasure-Arousal-Dominance).
        
        По модели Russell Circumplex и PAD.
        """
        emotion_pad = {
            "joy": (0.8, 0.6, 0.5),       # Позитив, активация, контроль
            "anger": (-0.6, 0.8, 0.6),    # Негатив, высокая активация, контроль
            "sadness": (-0.7, 0.2, -0.3), # Негатив, низкая активация, низкий контроль
            "fear": (-0.8, 0.7, -0.5),    # Негатив, активация, нет контроля
            "surprise": (0.2, 0.8, 0.0),  # Слабый позитив, высокая активация
            "disgust": (-0.5, 0.4, 0.2),  # Негатив, средняя активация
            "neutral": (0.0, 0.3, 0.0),   # Нейтрально
        }
        
        return emotion_pad.get(emotion, (0.0, 0.3, 0.0))


class FalseBeliefHandler:
    """
    Понимание ложных убеждений (False Belief Test).
    Классический тест: Sally-Anne.
    """
    
    def __init__(self):
        self.test_scenarios = []
    
    def sally_anne_test(self, scenario: str) -> Dict:
        """
        Sally-Anne тест на понимание ложных убеждений.
        
        Сценарий:
        1. Sally кладет шарик в корзину
        2. Sally уходит
        3. Anne перекладывает шарик в коробку
        4. Sally возвращается
        
        Вопрос: Где Sally будет искать шарик?
        Правильный ответ: В корзине (она не знает что Anne переложила)
        
        Args:
            scenario: Описание сценария
        
        Returns:
            Результат теста
        """
        scenario_lower = scenario.lower()
        
        # Простой паттерн матчинг для Sally-Anne
        if "sally" in scenario_lower and "anne" in scenario_lower:
            if "корзин" in scenario_lower and "коробк" in scenario_lower:
                # Классический Sally-Anne
                
                # Вопрос: где будет искать?
                if "где" in scenario_lower and "искать" in scenario_lower:
                    # Правильный ответ: в корзине (ложное убеждение Sally)
                    return {
                        "answer": "В корзине",
                        "reasoning": "Sally не знает что Anne переложила шарик, поэтому будет искать там где оставила",
                        "understands_false_belief": True
                    }
        
        return {
            "answer": "Недостаточно информации",
            "reasoning": "Не распознан как Sally-Anne тест",
            "understands_false_belief": False
        }
    
    def track_user_knowledge(self, fact: str, user_knows: bool):
        """
        Отслеживать что пользователь знает/не знает.
        
        Args:
            fact: Факт
            user_knows: Знает ли пользователь этот факт
        """
        self.test_scenarios.append({
            "fact": fact,
            "user_knows": user_knows,
            "timestamp": time()
        })


class EmpathySystem:
    """
    Система эмпатии - эмоциональное заражение и сочувствие.
    """
    
    def __init__(self):
        self.empathy_strength = 0.5  # Насколько сильно агент эмпатирует (0-1)
    
    def empathize(
        self,
        user_mental_state: MentalStateEstimate,
        agent_state
    ) -> Dict[str, float]:
        """
        Эмпатия: "заразиться" эмоцией пользователя.
        
        Args:
            user_mental_state: Оценка состояния пользователя
            agent_state: Текущее состояние агента
        
        Returns:
            Deltas для состояния агента
        """
        # Эмоциональное заражение (emotional contagion)
        # Агент частично "ловит" эмоцию пользователя
        
        deltas = {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0
        }
        
        # Если пользователь в негативе - агент тоже немного
        if user_mental_state.valence < -0.5:
            delta_valence = user_mental_state.valence * self.empathy_strength * 0.3
            deltas["valence"] = delta_valence
            print(f"[ToM] Эмпатия: пользователь расстроен, агент разделяет (valence {delta_valence:+.2f})")
        
        # Если пользователь в радости - агент тоже
        if user_mental_state.valence > 0.5:
            delta_valence = user_mental_state.valence * self.empathy_strength * 0.2
            deltas["valence"] = delta_valence
            print(f"[ToM] Эмпатия: пользователь рад, агент разделяет (valence {delta_valence:+.2f})")
        
        # Высокое возбуждение пользователя → агент тоже активируется
        if user_mental_state.arousal > 0.7:
            delta_arousal = user_mental_state.arousal * self.empathy_strength * 0.2
            deltas["arousal"] = delta_arousal
            print(f"[ToM] Эмпатия: пользователь возбужден, агент активируется (arousal {delta_arousal:+.2f})")
        
        return deltas
    
    def generate_empathic_response(self, user_mental_state: MentalStateEstimate) -> Optional[str]:
        """
        Сгенерировать эмпатичный ответ.
        
        Args:
            user_mental_state: Состояние пользователя
        
        Returns:
            Эмпатичная фраза или None
        """
        emotion = user_mental_state.emotion
        
        # Фразы сочувствия для разных эмоций
        empathy_phrases = {
            "anger": [
                "Понимаю, ты расстроен.",
                "Вижу, что тебя что-то задело.",
                "Чувствую твое раздражение.",
            ],
            "sadness": [
                "Мне жаль что ты расстроен.",
                "Понимаю, это непросто.",
                "Разделяю твою печаль.",
            ],
            "fear": [
                "Понимаю твое беспокойство.",
                "Это действительно может тревожить.",
                "Твои переживания понятны.",
            ],
            "joy": [
                "Рад разделить твою радость!",
                "Здорово что ты в хорошем настроении!",
                "Разделяю твой энтузиазм!",
            ],
        }
        
        if emotion in empathy_phrases:
            return random.choice(empathy_phrases[emotion])
        
        return None


class TheoryOfMindSystem:
    """
    Интегрированная система Theory of Mind.
    Объединяет все компоненты ToM.
    """
    
    def __init__(self, emotion_system, self_model):
        self.belief_tracker = BeliefTracker()
        self.intention_recognizer = IntentionRecognizer()
        self.mental_state_simulator = MentalStateSimulator(emotion_system)
        self.false_belief_handler = FalseBeliefHandler()
        self.empathy_system = EmpathySystem()
        
        self.self_model = self_model
        
        # История ToM событий
        self.tom_history = []
    
    def process_user_input(
        self,
        user_text: str,
        agent_state,
        dialog_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Обработать ввод пользователя через ToM.
        
        Args:
            user_text: Текст пользователя
            agent_state: Состояние агента
            dialog_history: История диалога
        
        Returns:
            Результат ToM анализа
        """
        # 1. Обновить убеждения
        beliefs = self.belief_tracker.update_beliefs(user_text)
        
        # 2. Распознать намерение
        intention = self.intention_recognizer.recognize(user_text, dialog_history)
        
        # 3. Симулировать ментальное состояние пользователя
        user_mental_state = self.mental_state_simulator.simulate(user_text)
        
        # 4. Эмпатия - влияние на агента
        empathy_deltas = self.empathy_system.empathize(user_mental_state, agent_state)
        
        # Применить empathy deltas к состоянию агента
        if empathy_deltas["valence"] != 0:
            agent_state.valence += empathy_deltas["valence"]
            agent_state.valence = max(-1.0, min(1.0, agent_state.valence))
        
        if empathy_deltas["arousal"] != 0:
            agent_state.arousal += empathy_deltas["arousal"]
            agent_state.arousal = max(0.0, min(1.0, agent_state.arousal))
        
        # 5. Сгенерировать эмпатичный ответ (опционально)
        empathic_phrase = self.empathy_system.generate_empathic_response(user_mental_state)
        
        # Собрать результат
        result = {
            "beliefs": beliefs,
            "intention": intention,
            "user_mental_state": user_mental_state,
            "empathy_deltas": empathy_deltas,
            "empathic_phrase": empathic_phrase,
            "timestamp": time()
        }
        
        # Сохранить в историю
        self.tom_history.append(result)
        if len(self.tom_history) > 100:
            self.tom_history.pop(0)
        
        return result
    
    def get_tom_summary(self) -> str:
        """Получить сводку ToM для использования в промптах"""
        # Последние убеждения
        beliefs = self.belief_tracker.get_all_beliefs()
        beliefs_text = "\n".join([
            f"- {b.topic}: {b.content} (confidence: {b.confidence:.2f})"
            for b in beliefs[-3:]  # Последние 3
        ]) if beliefs else "Нет данных об убеждениях"
        
        # Последнее намерение
        if self.tom_history:
            last_intention = self.tom_history[-1]["intention"]
            intention_text = f"{last_intention.intention_type} (confidence: {last_intention.confidence:.2f})"
        else:
            intention_text = "Неизвестно"
        
        # Последнее состояние
        if self.tom_history:
            last_state = self.tom_history[-1]["user_mental_state"]
            state_text = f"{last_state.emotion} (valence: {last_state.valence:+.2f}, arousal: {last_state.arousal:.2f})"
        else:
            state_text = "Неизвестно"
        
        summary = f"""Theory of Mind (понимание пользователя):
- Убеждения пользователя:
{beliefs_text}
- Намерение: {intention_text}
- Эмоциональное состояние: {state_text}
"""
        
        return summary
    
    def get_stats(self) -> Dict:
        """Статистика ToM"""
        return {
            "total_beliefs": len(self.belief_tracker.beliefs),
            "tom_events": len(self.tom_history),
            "empathy_strength": self.empathy_system.empathy_strength,
        }

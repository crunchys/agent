from mental_state import MentalState
from systems import EmotionSystem, PredictionErrorSystem, AttentionSystem, FutureExpectationSystem
from memory_classes import VectorMemory, EpisodicMemory, MetaReflection
from model_classes import SelfModel, OtherModel
from generators import ThoughtGenerator, ResponseGenerator
from agent_memory import PersistentMemory  # Твой модуль
from utils import load_model_and_tokenizer
from inputimeout import inputimeout, TimeoutOccurred
import random
from time import time
import torch
from typing import List, Dict, Optional
from collections import Counter

class Agent:
    def __init__(self, hf_token=None):
        # Загружаем модель один раз
        model, tokenizer = load_model_and_tokenizer(hf_token=hf_token)
        
        self.state = MentalState.initial()
        self.emotion = EmotionSystem()
        self.attention = AttentionSystem()
        self.prediction = PredictionErrorSystem()
        self.memory = PersistentMemory("memory.json")  # Эпизодическая память
        self.vector_memory = VectorMemory()  # Векторная память (Faiss)
        self.meta = MetaReflection(model=model, tokenizer=tokenizer)
        self.thought_gen = ThoughtGenerator(model=model, tokenizer=tokenizer)
        self.response_gen = ResponseGenerator(model=model, tokenizer=tokenizer)
        self.self_model = SelfModel()
        self.other_model = OtherModel()  # Новая модель других (пользователя)
        self.future = FutureExpectationSystem()
        self.last_thought: Optional[str] = None
        self.dialog_history: List[Dict[str, str]] = []  # История диалога
        self.last_self_evaluation: str = ""  # Последняя самооценка

    def step(self, stimuli: List[Dict]):
        prediction_errors = []
        curiosity = 0.0

        for s in stimuli:
            pe = self.prediction.compute(s)
            s["prediction_error"] = pe
            prediction_errors.append(pe)
            s["curiosity"] = self.future.curiosity(s["content"])
            curiosity += s["curiosity"]

        avg_prediction_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0
        avg_curiosity = curiosity / len(stimuli) if stimuli else 0.0

        for s in stimuli:
            self.emotion.apply_stimulus(self.state, s)

        self.state.focus = self.attention.select_focus(stimuli)

        last_events = self.memory.recent(5)

        # Обновляем страх смерти, если нет стимулов (тишина = угроза)
        if not stimuli:
            self.emotion.update_existence_threat(self.state, 0.1)  # Таймаут усиливает threat

        thought = self.thought_gen.generate_thought(
            self.state.focus,
            self.state.arousal,
            self.state.valence,
            avg_prediction_error,
            last_events,
            self.self_model,
            avg_curiosity,
            self.vector_memory,  # Передаем vector_memory для поиска
            self.dialog_history,  # Передаем историю диалога
            self.state.existence_threat,  # Передаем threat
            self.last_self_evaluation  # Передаем последнюю самооценку
        )

        event = {
            "type": "experience",
            "time": round(time(), 2),
            "focus": self.state.focus,
            "arousal": round(self.state.arousal, 3),
            "valence": round(self.state.valence, 3),
            "existence_threat": round(self.state.existence_threat, 3),  # Сохраняем threat
            "prediction_error": round(avg_prediction_error, 3),
            "curiosity": round(avg_curiosity, 3),
            "thought": thought,
        }

        self.memory.store(event)
        self.vector_memory.add_event(event)  # Добавляем в векторную память

        meta_ref = self.meta.reflect(self.memory)
        if meta_ref:
            self.memory.store(meta_ref)
            self.self_model.reflect(self.state, self.memory, new_lesson=meta_ref.get("lesson"))

        self.self_model.reflect(self.state, self.memory)

        for s in stimuli:
            self.future.update(s["content"], self.state.valence, self.state.arousal)

        self.emotion.decay(self.state)
        self.state.timestamp = time()

        self.last_thought = thought
        return thought

    def respond(self, user_text: str) -> str:
        if self.last_thought is None:
            self.step([])  # Генерация начальной мысли, если нужно

        # Обновляем модель других перед ответом
        self.other_model.update_traits(user_text, self.response_gen.model, self.response_gen.tokenizer)
        predicted_user_behavior = self.other_model.predict_behavior(self.dialog_history, self.response_gen.model, self.response_gen.tokenizer)

        response = self.response_gen.generate(
            self.last_thought, user_text, self.self_model, self.dialog_history,
            self.state.valence, self.state.arousal, self.state.existence_threat,  # Эмоции
            self.other_model  # Передаем модель других
        )
        
        # Добавляем в историю диалога
        self.dialog_history.append({"role": "user", "content": user_text})
        self.dialog_history.append({"role": "assistant", "content": response})
        
        # Ограничиваем историю (последние 20 реплик)
        if len(self.dialog_history) > 20:
            self.dialog_history = self.dialog_history[-20:]
        
        # Реакция на успех/неудачу
        is_success = self.state.valence > 0.2  # Простая эвристика
        self.emotion.apply_success_failure(self.state, is_success)
        
        # Метапознание: Оцениваем ответ
        action_desc = f"Ответил на '{user_text[:20]}...' с текстом '{response[:20]}...'."
        self.last_self_evaluation = self.self_model.evaluate_action(action_desc, self.state.valence, self.response_gen.model, self.response_gen.tokenizer)
        
        return response

    def generate_initiative(self) -> Dict[str, float | str]:
        """
        Создаёт осознанную инициативу агента, основанную
        на текущем состоянии, памяти и любопытстве.
        """

        # 1. Получаем контекст: последние события
        last_events = self.memory.recent(5)
        last_focuses = [e["focus"] for e in last_events if e.get("focus")]

        # 2. Определяем любопытство и фокус
        if not last_focuses:
            focus = "самоанализ"
        else:
            focus = random.choice(last_focuses)  # Или более умно выбрать

        stimuli = [{
            "content": "Внутренняя инициатива",
            "intensity": random.uniform(0.1, 0.3),
            "valence": random.uniform(-0.1, 0.1),
            "salience": random.uniform(0.4, 0.6),
        }]

        thought = self.step(stimuli)

        # Генерируем "инициативный" ответ (как будто пользователь спросил "что дальше?")
        initiative_response = self.respond("Что ты думаешь дальше?")

        return {
            "thought": thought,
            "response": initiative_response
        }


if __name__ == "__main__":
    agent = Agent()  # Здесь модель загрузится один раз

    print("Агент готов. Введите сообщение (или 'exit' для выхода).")

    while True:
        try:
            user_input = inputimeout(prompt="Вы: ", timeout=60)
        except TimeoutOccurred:
            print("Таймаут. Генерирую спонтанную мысль...")
            # Таймаут усиливает страх смерти
            agent.emotion.update_existence_threat(agent.state, 0.15)
            initiative = agent.generate_initiative()
            print(f"Мысль агента: {initiative['thought']}")
            print(f"Агент (инициатива): {initiative['response']}")
            continue

        if user_input.lower() == 'exit':
            break

        stimuli = [
            {
                "content": user_input,
                "intensity": random.uniform(0.1, 0.5),
                "valence": random.uniform(-0.2, 0.2),
                "salience": random.uniform(0.3, 0.8),
            }
        ]

        thought = agent.step(stimuli)
        print(f"Мысль агента: {thought}")

        response = agent.respond(user_input)
        print(f"Агент: {response}")

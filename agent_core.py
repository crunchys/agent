from mental_state import MentalState
from systems import EmotionSystem, PredictionErrorSystem, AttentionSystem, FutureExpectationSystem
from memory_classes import VectorMemory, EpisodicMemory, MetaReflection
from model_classes import SelfModel, OtherModel
from generators import ThoughtGenerator, ResponseGenerator
from agent_memory import PersistentMemory
from utils import load_model_and_tokenizer
from planning import Planner, Goal
from inputimeout import inputimeout, TimeoutOccurred
from typing import List, Dict, Optional
import random
from time import time
import torch


class Agent:
    def __init__(self, hf_token=None):
        model, tokenizer = load_model_and_tokenizer(hf_token=hf_token)
        
        self.state = MentalState.initial()
        self.emotion = EmotionSystem()
        self.attention = AttentionSystem()
        self.prediction = PredictionErrorSystem()
        self.memory = PersistentMemory("memory.json")
        self.vector_memory = VectorMemory()
        self.meta = MetaReflection()  # Без model/tokenizer
        self.thought_gen = ThoughtGenerator(model=model, tokenizer=tokenizer)
        self.response_gen = ResponseGenerator(model=model, tokenizer=tokenizer)
        self.self_model = SelfModel()
        self.other_model = OtherModel()
        self.future = FutureExpectationSystem()
        self.planner = Planner(self.self_model, self.future)  # Новый planner
        self.last_thought: Optional[str] = None
        self.dialog_history: List[Dict[str, str]] = []

    def step(self, stimuli: List[Dict]):
        # 1. Prediction Error
        prediction_errors = [self.prediction.compute(s) for s in stimuli]
        avg_prediction_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0

        # 2. Attention
        self.state.focus = self.attention.select_focus(stimuli)

        # 3. Emotion update
        for s in stimuli:
            self.emotion.apply_stimulus(self.state, s)
        self.emotion.decay(self.state)

        # 4. Tool use / Grounding
        if stimuli:
            query = stimuli[0]["content"]
            relevant_memories = self.vector_memory.search(query, k=3)  # Grounding в памяти

        # 5. Self-model update
        last_events = self.memory.recent(5)
        self.self_model.reflect(self.state, self.memory)

        # 6. Meta reflection (правила)
        meta_ref = self.meta.reflect(self.memory)
        if meta_ref:
            self.memory.store(meta_ref)
            self.self_model.reflect(self.state, self.memory, new_lesson=meta_ref.get("lesson"))

        # 7. Planning (non-verbal)
        new_goal = self.planner.form_goal(self.state, self.future.curiosity(stimuli[0]["content"] if stimuli else ""), self.state.existence_threat, avg_prediction_error)
        if new_goal:
            self.planner.goals.append(new_goal)
        
        current_action = self.planner.choose_action()
        is_success = self.state.valence > 0.2
        self.planner.update(is_success)

        # 8. Vocalization (LLM только здесь)
        state_summary = (
            f"Фокус: {self.state.focus}\n"
            f"Состояние: arousal={self.state.arousal:.2f}, valence={self.state.valence:.2f}, threat={self.state.existence_threat:.2f}\n"
            f"Текущая цель: {self.planner.goals[0].description if self.planner.goals else 'нет'}\n"
            f"Текущее действие: {current_action or 'нет'}\n"
            f"Любопытство: {self.future.curiosity(stimuli[0]["content"] if stimuli else 0)}\n"
        )

        thought = self.thought_gen.generate_thought(state_summary)  # Промпт с summary

        event = {
            "type": "experience",
            "time": round(time(), 2),
            "focus": self.state.focus,
            "arousal": round(self.state.arousal, 3),
            "valence": round(self.state.valence, 3),
            "existence_threat": round(self.state.existence_threat, 3),
            "prediction_error": round(avg_prediction_error, 3),
            "thought": thought,
        }

        self.memory.store(event)
        self.vector_memory.add_event(event)

        self.last_thought = thought
        return thought


    def respond(self, user_text: str) -> str:
        if self.last_thought is None:
            self.step([])

        # Обновляем модель других
        self.other_model.update_traits(user_text, self.response_gen.model, self.response_gen.tokenizer)
        predicted_user_behavior = self.other_model.predict_behavior(self.dialog_history, self.response_gen.model, self.response_gen.tokenizer)
        current_action = self.planner.get_current_action()

        response = self.response_gen.generate(
            self.last_thought, user_text, self.self_model, self.dialog_history,
            self.state.valence, self.state.arousal, self.state.existence_threat,
            self.other_model, self.current_curiosity, current_action or "" # Добавляем действие
        )
        
        # Добавляем в историю диалога
        self.dialog_history.append({"role": "user", "content": user_text})
        self.dialog_history.append({"role": "assistant", "content": response})
        
        if len(self.dialog_history) > 20:
            self.dialog_history = self.dialog_history[-20:]
        
        # Реакция на успех/неудачу
        is_success = self.state.valence > 0.2
        self.emotion.apply_success_failure(self.state, is_success)
        
        # Метапознание
        action_desc = f"Ответил на '{user_text[:20]}...' с текстом '{response[:20]}...'."
        self.last_self_evaluation = self.self_model.evaluate_action(action_desc, self.state.valence, self.response_gen.model, self.response_gen.tokenizer)

        return response

    def generate_initiative(self) -> Dict[str, float | str]:
        last_events = self.memory.recent(5)
        last_focuses = [e["focus"] for e in last_events if e.get("focus")]

        if not last_focuses:
            focus = "самоанализ"
        else:
            focus = random.choice(last_focuses)

        stimuli = [{
            "content": "Внутренняя инициатива",
            "intensity": random.uniform(0.1, 0.3),
            "valence": random.uniform(-0.1, 0.1),
            "salience": random.uniform(0.4, 0.6),
        }]

        thought = self.step(stimuli)

        current_action = self.planner.get_current_action()
        initiative_response = self.respond("Что ты думаешь дальше?", current_action=current_action)

        return {
            "thought": thought,
            "response": initiative_response
        }


if __name__ == "__main__":
    agent = Agent()

    print("Агент готов. Введите сообщение (или 'exit' для выхода).")

    while True:
        try:
            user_input = inputimeout(prompt="Вы: ", timeout=60)
        except TimeoutOccurred:
            print("Таймаут. Генерирую спонтанную мысль...")
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

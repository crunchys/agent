from mental_state import MentalState
from systems import EmotionSystem, PredictionErrorSystem, AttentionSystem, FutureExpectationSystem
from memory_classes import VectorMemory, EpisodicMemory, MetaReflection
from model_classes import SelfModel, OtherModel
from generators import ThoughtGenerator, ResponseGenerator
from agent_memory import PersistentMemory
from planning import Planner  # Добавлен импорт
from utils import load_model_and_tokenizer
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
        self.meta = MetaReflection()
        self.thought_gen = ThoughtGenerator(model=model, tokenizer=tokenizer)
        self.response_gen = ResponseGenerator(model=model, tokenizer=tokenizer)
        self.self_model = SelfModel()
        self.other_model = OtherModel()
        self.future = FutureExpectationSystem()
        self.planner = Planner(self.self_model, self.future)  # Теперь Planner импортирован
        self.last_thought: Optional[str] = None
        self.dialog_history: List[Dict[str, str]] = []
        self.current_curiosity: float = 0.0

    def step(self, stimuli: List[Dict]):
        try:
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

            self.current_curiosity = avg_curiosity

            for s in stimuli:
                self.emotion.apply_stimulus(self.state, s)

            self.state.focus = self.attention.select_focus(stimuli)

            last_events = self.memory.recent(5)

            if not stimuli:
                self.emotion.update_existence_threat(self.state, 0.1)

            content_for_curiosity = stimuli[0]["content"] if stimuli else ""
            curiosity_value = self.future.curiosity(content_for_curiosity)

            # Планирование
            if random.random() < 0.4 or not self.planner.goals:
                new_goal = self.planner.form_goal(self.state, avg_curiosity, self.state.existence_threat, avg_prediction_error)
                if new_goal:
                    self.planner.goals.append(new_goal)
                    print(f"[ПЛАНИРОВАНИЕ] Новая цель: {new_goal.description}")

            if stimuli:
                is_success = self.state.valence > 0
                for goal in self.planner.goals[:]:
                    if goal.status == "active":
                        self.planner.update_progress(goal, is_success)
                        if goal.status != "active":
                            print(f"[ПЛАНИРОВАНИЕ] Цель '{goal.description}' {goal.status}")

            self.planner.goals = [g for g in self.planner.goals if g.status == "active"][-5:]

            current_action = self.planner.get_current_action() or "нет"

            # Безопасно: если goals пустой — 'нет'
            current_goal_desc = self.planner.goals[0].description if self.planner.goals else 'нет'

            state_summary = (
                f"Фокус: {self.state.focus or 'нет фокуса'}\n"
                f"Состояние: arousal={self.state.arousal:.2f}, valence={self.state.valence:.2f}, threat={self.state.existence_threat:.2f}\n"
                f"Текущая цель: {current_goal_desc}\n"
                f"Текущее действие: {current_action}\n"
                f"Любопытство: {curiosity_value:.2f}\n"
            )

            thought = self.thought_gen.generate_thought(state_summary)

            event = {
                "type": "experience",
                "time": round(time(), 2),
                "focus": self.state.focus,
                "arousal": round(self.state.arousal, 3),
                "valence": round(self.state.valence, 3),
                "existence_threat": round(self.state.existence_threat, 3),
                "prediction_error": round(avg_prediction_error, 3),
                "curiosity": round(avg_curiosity, 3),
                "thought": thought,
            }

            self.memory.store(event)
            self.vector_memory.add_event(event)

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
        except Exception as e:
            print(f"Ошибка в step: {e}")
            import traceback
            traceback.print_exc()
            return "Ошибка в обработке"

    def respond(self, user_text: str) -> str:
        try:
            if self.last_thought is None:
                self.step([])

            self.other_model.update_traits(user_text, self.response_gen.model, self.response_gen.tokenizer)
            predicted_user_behavior = self.other_model.predict_behavior(self.dialog_history, self.response_gen.model, self.response_gen.tokenizer)

            current_action = self.planner.get_current_action() or ""

            response = self.response_gen.generate(
                self.last_thought, user_text, self.self_model, self.dialog_history,
                self.state.valence, self.state.arousal, self.state.existence_threat,
                self.other_model, self.current_curiosity, current_action
            )
            
            self.dialog_history.append({"role": "user", "content": user_text})
            self.dialog_history.append({"role": "assistant", "content": response})
            
            if len(self.dialog_history) > 20:
                self.dialog_history = self.dialog_history[-20:]
            
            is_success = self.state.valence > 0.2
            self.emotion.apply_success_failure(self.state, is_success)
            
            action_desc = f"Ответил на '{user_text[:20]}...' с текстом '{response[:20]}...'."
            self.last_self_evaluation = self.self_model.evaluate_action(action_desc, self.state.valence, self.response_gen.model, self.response_gen.tokenizer)
            
            return response
        except Exception as e:
            print(f"Ошибка в respond: {e}")
            import traceback
            traceback.print_exc()
            return "Извини, произошла ошибка."

    def generate_initiative(self) -> Dict[str, float | str]:
        try:
            last_events = self.memory.recent(5)
            last_focuses = [e["focus"] for e in last_events if e.get("focus")]

            focus = random.choice(last_focuses) if last_focuses else "самоанализ"

            stimuli = [{
                "content": "Внутренняя инициатива",
                "intensity": random.uniform(0.1, 0.3),
                "valence": random.uniform(-0.1, 0.1),
                "salience": random.uniform(0.4, 0.6),
            }]

            thought = self.step(stimuli)

            initiative_response = self.respond("Что ты думаешь дальше?")

            return {
                "thought": thought,
                "response": initiative_response
            }
        except Exception as e:
            print(f"Ошибка в generate_initiative: {e}")
            return {"thought": "Ошибка", "response": "Извини, что-то пошло не так."}


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

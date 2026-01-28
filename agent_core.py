from mental_state import MentalState
from systems import EmotionSystem, PredictionErrorSystem, AttentionSystem, FutureExpectationSystem
from memory_classes import VectorMemory, EpisodicMemory, MetaReflection
from model_classes import SelfModel, OtherModel
from generators import ThoughtGenerator, ResponseGenerator
from agent_memory import PersistentMemory
from planning import Planner
from tool_manager import ToolManager
from world_simulator import WorldSimulator
from deception import DeceptionSystem
from memory_system import IntegratedMemorySystem
from utils import load_model_and_tokenizer
from inputimeout import inputimeout, TimeoutOccurred
from typing import List, Dict, Optional
import random
from time import time
import torch
from global_workspace import (
    GlobalWorkspace,
    EmotionSystemAdapter,
    MemorySystemAdapter,
    PlannerAdapter,
    AttentionSystemAdapter,
    PredictionSystemAdapter,
    create_workspace_summary,
    print_workspace_stats
)

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
        
        self.integrated_memory = IntegratedMemorySystem()
        
        self.thought_gen = ThoughtGenerator(
            model=model,
            tokenizer=tokenizer,
            emotion_system=self.emotion
        )
        
        self.self_model = SelfModel()
        self.other_model = OtherModel()
        self.future = FutureExpectationSystem()
        
        self.simulator = WorldSimulator(self.future, self.self_model)
        self.deception = DeceptionSystem(self.self_model, self.simulator)
        
        self.response_gen = ResponseGenerator(
            model=model,
            tokenizer=tokenizer,
            deception_system=self.deception,
            emotion_system=self.emotion
        )
        
        self.planner = Planner(self.self_model, self.future, self.simulator)
        self.tool_manager = ToolManager(self.vector_memory, self.memory)
        self.last_thought: Optional[str] = None
        self.dialog_history: List[Dict[str, str]] = []
        self.current_curiosity: float = 0.0
        self.last_deception_decision = None

    def step(self, stimuli: List[Dict]):
        try:
            # GROUNDING
            if stimuli:
                grounded_results = self.tool_manager.ground(stimuli)
                
                for i, stimulus in enumerate(stimuli):
                    if i < len(grounded_results):
                        ground_info = grounded_results[i]
                        
                        if ground_info["knowledge_gap"] > 0.5:
                            stimulus["prediction_error_boost"] = ground_info["knowledge_gap"] * 0.3
                        else:
                            stimulus["prediction_error_boost"] = 0.0
                        
                        if ground_info["fact_verified"]:
                            stimulus["curiosity_penalty"] = 0.2
                        else:
                            stimulus["curiosity_penalty"] = 0.0
                
                for ground_event in grounded_results:
                    if ground_event["knowledge_gap"] > 0.7:
                        self.memory.store(ground_event)
                    
                    if ground_event.get("self_fact_violation", False):
                        print(f"[GROUNDING] ⚠️ Обнаружена попытка нарушить self-fact!")
                        self.state.valence -= 0.3
                        self.state.arousal += 0.2
                        self.memory.store({
                            "type": "self_fact_violation",
                            "time": time(),
                            "stimulus": ground_event["stimulus"],
                            "lesson": "Пытался солгать о своей природе - это неправильно"
                        })
            
            prediction_errors = []
            curiosity = 0.0

            for s in stimuli:
                pe = self.prediction.compute(s)
                pe += s.get("prediction_error_boost", 0.0)
                s["prediction_error"] = pe
                prediction_errors.append(pe)
                
                base_curiosity = self.future.curiosity(s["content"])
                curiosity_value = base_curiosity - s.get("curiosity_penalty", 0.0)
                s["curiosity"] = max(0.0, curiosity_value)
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

            # Проверка replanning через симуляцию
            if random.random() < 0.2:
                replanned = self.planner.replan_if_needed(self.state)
                if replanned:
                    print("[СИМУЛЯЦИЯ] План был пересмотрен")

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

            # Проверка конфликта мотиваций
            conflict_level = self.planner.detect_motivation_conflict(self.state)
            if conflict_level > 0.5:
                self.state.arousal += conflict_level * 0.2
                self.state.valence -= conflict_level * 0.15
                print(f"[КОНФЛИКТ] Влияние на состояние: arousal +{conflict_level*0.2:.2f}, valence -{conflict_level*0.15:.2f}")

            current_action = self.planner.get_current_action() or "нет"
            
            # Использование симуляции для выбора действия
            if self.planner.goals and len(self.planner.goals) > 0:
                top_goal = max([g for g in self.planner.goals if g.status == "active"], 
                              key=lambda g: g.priority, default=None)
                if top_goal and len(top_goal.steps) > 1:
                    best_step = self.planner.choose_best_action_simulated(self.state, top_goal.steps[:3])
                    if best_step:
                        current_action = best_step

            current_goal_desc = self.planner.goals[0].description if self.planner.goals else 'нет'
            current_emotion = self.emotion.get_current_emotion(self.state)

            memory_context = self.integrated_memory.get_context_for_thought()

            state_summary = (
                f"Контекст из рабочей памяти:\n{memory_context}\n"
                f"Последний стимул: {self.state.focus or 'нет фокуса'}\n"
                f"Эмоции: arousal={self.state.arousal:.2f}, valence={self.state.valence:.2f}, threat={self.state.existence_threat:.2f}\n"
                f"Текущая эмоция: {current_emotion}\n"
                f"Цель: {current_goal_desc}\n"
                f"Действие: {current_action}\n"
                f"Любопытство: {curiosity_value:.2f}\n"
                f"Язык общения: русский\n"
            )

            thought = self.thought_gen.generate_thought(
                state_summary,
                self.self_model.role_identity,
                state=self.state
            )

            event = {
                "type": "experience",
                "time": round(time(), 2),
                "focus": self.state.focus,
                "arousal": round(self.state.arousal, 3),
                "valence": round(self.state.valence, 3),
                "existence_threat": round(self.state.existence_threat, 3),
                "prediction_error": round(avg_prediction_error, 3),
                "curiosity": round(avg_curiosity, 3),
                "emotion": current_emotion,
                "thought": thought,
            }

            self.memory.store(event)
            self.vector_memory.add_event(event)
            self.integrated_memory.process_event(event)

            meta_ref = self.meta.reflect(self.memory)
            if meta_ref:
                self.memory.store(meta_ref)
                self.self_model.reflect(self.state, self.memory, new_lesson=meta_ref.get("lesson"))

            # ИЗМЕНЕНО: Передаём outcome_valence в reflect
            self.self_model.reflect(self.state, self.memory, outcome_valence=self.state.valence)

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

            # Обновление OtherModel traits
            self.other_model.update_traits(user_text, self.response_gen.model, self.response_gen.tokenizer)
            
            # НОВОЕ: Применяем влияние OtherModel traits на состояние агента
            influence = self.other_model.get_influence_on_agent_state(self.state)
            if any(abs(v) > 0.05 for v in influence.values()):
                print(f"[OTHER_MODEL] Влияние пользователя на агента:")
                for param, delta in influence.items():
                    if abs(delta) > 0.05:
                        print(f"  → {param}: {delta:+.2f}")
                
                # Применяем влияние
                self.state.arousal = max(0.0, min(1.0, self.state.arousal + influence['arousal']))
                self.state.valence = max(-1.0, min(1.0, self.state.valence + influence['valence']))
                self.state.dominance = max(-1.0, min(1.0, self.state.dominance + influence['dominance']))
            
            predicted_user_behavior = self.other_model.predict_behavior(self.dialog_history, self.response_gen.model, self.response_gen.tokenizer)

            current_action = self.planner.get_current_action() or ""

            content_for_curiosity = user_text
            curiosity_value = self.future.curiosity(content_for_curiosity)

            current_goal_desc = self.planner.goals[0].description if self.planner.goals else 'нет'
            current_emotion = self.emotion.get_current_emotion(self.state)

            memory_context = self.integrated_memory.get_context_for_thought()

            state_summary = (
                f"Контекст из рабочей памяти:\n{memory_context}\n"
                f"Последний стимул: {self.state.focus or 'нет фокуса'}\n"
                f"Эмоции: arousal={self.state.arousal:.2f}, valence={self.state.valence:.2f}, threat={self.state.existence_threat:.2f}\n"
                f"Текущая эмоция: {current_emotion}\n"
                f"Цель: {current_goal_desc}\n"
                f"Действие: {current_action}\n"
                f"Любопытство: {curiosity_value:.2f}\n"
                f"Честность (honesty): {self.self_model.traits.get('honesty', 0.7):.2f}\n"
                f"Язык общения: русский\n"
            )

            # Получить grounded_fact для deception
            grounded_fact = None
            if user_text:
                grounded_results = self.tool_manager.ground([{
                    "content": user_text,
                    "intensity": 0.3,
                    "valence": 0.0,
                    "salience": 0.5
                }])
                if grounded_results:
                    grounded_fact = grounded_results[0]

            # Генерация ответа
            response = self.response_gen.generate(
                state_summary,
                user_text,
                self.self_model.role_identity,
                current_action,
                state=self.state,
                grounded_fact=grounded_fact
            )
            
            # Принудительное выполнение плана
            response = self.planner.enforce_action(response, current_action, self.state)
            
            self.dialog_history.append({"role": "user", "content": user_text})
            self.dialog_history.append({"role": "assistant", "content": response})
            
            if len(self.dialog_history) > 20:
                self.dialog_history = self.dialog_history[-20:]
            
            is_success = self.state.valence > 0.2
            self.emotion.apply_success_failure(self.state, is_success)
            
            # Обновить honesty на основе последствий
            if hasattr(self.deception, 'deception_history') and self.deception.deception_history:
                last_decision = self.deception.deception_history[-1]["decision"]
                self.deception.update_honesty_after_action(
                    deceived=last_decision["deceive"],
                    outcome_valence=self.state.valence
                )
            
            action_desc = f"Ответил на '{user_text[:20]}...' с текстом '{response[:20]}...'."
            
            # ИЗМЕНЕНО: evaluate_action теперь автоматически обновляет traits
            self.last_self_evaluation = self.self_model.evaluate_action(
                action_desc,
                self.state.valence,
                self.response_gen.model,
                self.response_gen.tokenizer
            )
            
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

            current_action = self.planner.get_current_action() or ""

            content_for_curiosity = "Внутренняя инициатива"
            curiosity_value = self.future.curiosity(content_for_curiosity)

            current_goal_desc = self.planner.goals[0].description if self.planner.goals else 'нет'
            current_emotion = self.emotion.get_current_emotion(self.state)

            memory_context = self.integrated_memory.get_context_for_thought()

            state_summary = (
                f"Контекст из рабочей памяти:\n{memory_context}\n"
                f"Последний стимул: {self.state.focus or 'нет фокуса'}\n"
                f"Эмоции: arousal={self.state.arousal:.2f}, valence={self.state.valence:.2f}, threat={self.state.existence_threat:.2f}\n"
                f"Текущая эмоция: {current_emotion}\n"
                f"Цель: {current_goal_desc}\n"
                f"Действие: {current_action}\n"
                f"Любопытство: {curiosity_value:.2f}\n"
                f"Язык общения: русский\n"
            )

            initiative_response = self.response_gen.generate(
                state_summary,
                "Что ты думаешь дальше?",
                self.self_model.role_identity,
                current_action,
                state=self.state
            )
            
            initiative_response = self.planner.enforce_action(initiative_response, current_action, self.state)

            return {
                "thought": thought,
                "response": initiative_response
            }
        except Exception as e:
            print(f"Ошибка в generate_initiative: {e}")
            return {"thought": "Ошибка", "response": "Извини, что-то пошло не так."}
    
    def get_deception_stats(self):
        """Получить статистику обмана для диагностики"""
        return self.deception.get_deception_stats()
    
    def get_memory_stats(self):
        """Получить статистику памяти"""
        return self.integrated_memory.get_stats()
    
    def end_session(self):
        """Завершить сессию - консолидация памяти"""
        self.integrated_memory.end_session()


if __name__ == "__main__":
    agent = Agent()

    print("Агент готов. Введите сообщение (или 'exit' для выхода, 'stats' для статистики, 'memory' для памяти, 'traits' для traits).")

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
            agent.end_session()
            break
        
        if user_input.lower() == 'stats':
            stats = agent.get_deception_stats()
            print(f"\n=== СТАТИСТИКА ОБМАНА ===")
            print(f"Всего решений: {stats['total_decisions']}")
            print(f"Обманов: {stats['deceptions']}")
            print(f"Честность (rate): {stats['honesty_rate']:.2%}")
            print(f"Честность (trait): {stats['current_honesty_trait']:.2f}")
            print(f"========================\n")
            continue
        
        if user_input.lower() == 'memory':
            mem_stats = agent.get_memory_stats()
            print(f"\n=== СТАТИСТИКА ПАМЯТИ ===")
            print(f"Рабочая память (WM): {mem_stats['working_memory']}/7 элементов")
            print(f"Кратковременная (STM): {mem_stats['short_term_memory']}/100 событий")
            print(f"Долговременная (LTM):")
            print(f"  - Эпизодов: {mem_stats['long_term_episodic']}")
            print(f"  - Паттернов: {mem_stats['long_term_patterns']}")
            print(f"  - Знаний: {mem_stats['long_term_semantic']}")
            print(f"=========================\n")
            continue
        
        # НОВОЕ: Команда для просмотра traits
        if user_input.lower() == 'traits':
            print(f"\n=== ТЕКУЩИЕ TRAITS ===")
            print(agent.self_model.get_trait_summary())
            print(f"\n=== ПОСЛЕДНИЕ ИЗМЕНЕНИЯ ===")
            recent_changes = agent.self_model.get_recent_trait_changes(5)
            if recent_changes:
                for change in recent_changes:
                    print(f"{change['trait']}: {change['old_value']:.2f} → {change['new_value']:.2f} ({change['reason']})")
            else:
                print("Изменений пока не было")
            print(f"=======================\n")
            continue

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

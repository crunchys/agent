from dataclasses import dataclass
from time import time, sleep
from inputimeout import inputimeout, TimeoutOccurred
from typing import Optional, List, Dict
import random
from collections import Counter

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from agent_memory import PersistentMemory  # 📌 память из отдельного файла

# ======================================================
# Mental State
# ======================================================

@dataclass
class MentalState:
    arousal: float
    valence: float
    focus: Optional[str]
    timestamp: float

    @classmethod
    def initial(cls):
        return cls(arousal=0.3, valence=0.0, focus=None, timestamp=time())

# ======================================================
# Emotion System
# ======================================================

class EmotionSystem:
    def __init__(self, decay_rate: float = 0.05):
        self.decay_rate = decay_rate

    def apply_stimulus(self, state: MentalState, stimulus: Dict):
        state.arousal = min(1.0, state.arousal + stimulus["intensity"])
        state.valence = max(-1.0, min(1.0, state.valence + stimulus["valence"]))
        return state

    def decay(self, state: MentalState):
        state.arousal *= (1 - self.decay_rate)
        state.valence *= (1 - self.decay_rate)
        return state

# ======================================================
# Prediction Error System
# ======================================================

class PredictionErrorSystem:
    def __init__(self):
        self.history = Counter()

    def compute(self, stimulus: Dict) -> float:
        content = stimulus["content"]
        self.history[content] += 1

        total = sum(self.history.values())
        freq = self.history[content] / total if total > 0 else 0.0

        prediction_error = 1.0 - freq
        return round(prediction_error, 3)

# ======================================================
# Attention System
# ======================================================

class AttentionSystem:
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def select_focus(self, stimuli: List[Dict]) -> Optional[str]:
        if not stimuli:
            return None
        best = max(stimuli, key=lambda s: s["salience"])
        return best["content"] if best["salience"] >= self.threshold else None

# ======================================================
# Self Model
# ======================================================

@dataclass
class SelfModel:
    name: str = "Агент"
    traits: Dict[str, float] = None
    goals: List[str] = None

    def __post_init__(self):
        if self.traits is None:
            self.traits = {'смелость': 0.5, 'любопытство': 0.5}
        if self.goals is None:
            self.goals = []

    def reflect(self, state: MentalState, memory: PersistentMemory):
        recent = memory.recent_events(5)
        if recent:
            avg_valence = sum(e['valence'] for e in recent) / len(recent)
            self.traits['любопытство'] = min(1.0, max(0.0, 0.5 + avg_valence * 0.5))

# ======================================================
# Future Expectation System
# ======================================================

class FutureExpectationSystem:
    def __init__(self):
        self.expectations: Dict[str, Dict[str, float]] = {}

    def predict(self, content: str):
        if content not in self.expectations:
            self.expectations[content] = {'prob': 0.5, 'valence': 0.0, 'arousal': 0.3}
        return self.expectations[content]

    def update(self, content: str, actual_valence: float, actual_arousal: float):
        exp = self.predict(content)
        lr = 0.1
        exp['valence'] += lr * (actual_valence - exp['valence'])
        exp['arousal'] += lr * (actual_arousal - exp['arousal'])
        exp['prob'] = min(1.0, max(0.0, exp['prob'] + lr * (1 - exp['prob'])))

    def curiosity(self, content: str):
        exp = self.predict(content)
        return 1.0 - exp['prob']

# ======================================================
# Thought Generator
# ======================================================

class ThoughtGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", device="cuda", hf_token=None):
        print(f"🔄 Загружаю LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_name, token=hf_token, device_map="auto", torch_dtype=torch.float16
        )

        self.system_prompt = (
            "Ты — внутренняя мысль агента, поток размышлений.\n"
            "Это не отчёт и не объяснение, а живой внутренний диалог с сомнениями,\n"
            "с опорой на прошлый опыт и попыткой понять, что происходит.\n"
            "Мысль — законченная, 3–5 предложений.\n"
            "В конце обязательно добавь <END_THOUGHT>.\n"
            "Не используй формулы: «мне нужно», «я должен», «следует», «необходимо»."
        )

    def describe_affect(self, arousal: float, valence: float) -> str:
        if arousal < 0.2:
            arousal_desc = "спокойно"
        elif arousal < 0.5:
            arousal_desc = "настороженно"
        elif arousal < 0.8:
            arousal_desc = "активно"
        else:
            arousal_desc = "взволнованно"

        if valence < -0.5:
            valence_desc = "печально"
        elif valence < 0.0:
            valence_desc = "тревожно"
        elif valence < 0.5:
            valence_desc = "нейтрально"
        else:
            valence_desc = "радостно"

        return f"{arousal_desc}, {valence_desc}"

    def generate_thought(
        self,
        focus,
        arousal,
        valence,
        prediction_error,
        last_events,
        self_model,
        curiosity,
        contrast_signal=None
    ):
        events_summary = ", ".join(
            f"{e['focus']} (v:{e['valence']:.2f})" for e in last_events
        )
        affect_desc = self.describe_affect(arousal, valence)

        contrast_text = ""
        if contrast_signal:
            contrast_text = (
                "Ранее я думал иначе, и теперь внутри есть такое противоречие.\n"
            )

        user_prompt = (
            f"Фокус: {focus}\n"
            f"Состояние: {affect_desc}\n"
            f"Любопытство: {curiosity:.2f}\n"
            f"{contrast_text}"
            f"Прошлые события: {events_summary}\n"
            f"Черты личности: {self_model.traits}\n"
            f"Ошибка предсказания: {prediction_error:.2f}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        output_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
        )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        if "<END_THOUGHT>" in text:
            text = text.split("<END_THOUGHT>")[0].strip()

        return text + " <END_THOUGHT>"

# ======================================================
# Response Generator
# ======================================================

class ResponseGenerator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

        self.system_prompt = (
            "Ты — ответ агента собеседнику.\n"
            "Отвечай естественно, по‑человечески.\n"
            "Можно усомниться, можно уточнять.\n"
            "Ответ — 1–3 предложения."
        )

    def generate(self, thought, user_text, self_model):
        user_prompt = (
            f"Ты — {self_model.name}.\n"
            f"Собеседник сказал: «{user_text}».\n"
            f"Твоя внутренняя мысль:\n{thought}\n"
            "Сформулируй ответ."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        output_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

# ======================================================
# Agent
# ======================================================

class Agent:
    def __init__(self):
        self.state = MentalState.initial()
        self.emotion = EmotionSystem()
        self.attention = AttentionSystem()
        self.prediction = PredictionErrorSystem()
        self.memory = PersistentMemory("memory.json")
        self.memory.load()  # 🧠 загружаем старую память
        self.thought_gen = ThoughtGenerator()
        self.response_gen = ResponseGenerator(
            self.thought_gen.tokenizer, self.thought_gen.model
        )
        self.self_model = SelfModel()
        self.future = FutureExpectationSystem()
        self.last_thought = None

    def step(self, stimuli: List[Dict]):
        prediction_errors = []
        curiosity = 0.0

        for s in stimuli:
            pe = self.prediction.compute(s)
            s["prediction_error"] = pe
            prediction_errors.append(pe)

            s["salience"] += pe * 0.5
            s["intensity"] += pe * 0.3

            self.state = self.emotion.apply_stimulus(self.state, s)
            self.future.update(s["content"], s["valence"], s["intensity"])
            curiosity += self.future.curiosity(s["content"])

            if s.get("type") == "interaction":
                self.memory.update_object(
                    "пользователь",
                    properties={"last_message": s["content"], "time": time()},
                )

        curiosity /= len(stimuli) if stimuli else 1
        self.state.arousal = min(1.0, self.state.arousal + curiosity * 0.2)

        self.state.focus = self.attention.select_focus(stimuli)
        self.state = self.emotion.decay(self.state)
        self.state.timestamp = time()

        # сохраняем опыт
        self.memory.store_event({
            "type": "experience",
            "time": round(self.state.timestamp, 2),
            "focus": self.state.focus,
            "arousal": round(self.state.arousal, 3),
            "valence": round(self.state.valence, 3),
            "prediction_error": prediction_errors[0] if prediction_errors else 0.0,
        })

        self.memory.update_object(
            "агент",
            properties={"last_focus": self.state.focus, "arousal": self.state.arousal, "valence": self.state.valence},
        )

        self.self_model.reflect(self.state, self.memory)
        last_five = self.memory.recent_events(5)

        trend_valence = "нейтральный"
        if len(last_five) >= 2:
            delta = last_five[-1]["valence"] - last_five[0]["valence"]
            trend_valence = "повышающийся" if delta > 0 else "падающий" if delta < 0 else "стабильный"

        contrast_signal = None
        if self.last_thought and len(last_five) >= 2:
            shift = last_five[-1]["valence"] - last_five[-2]["valence"]
            if abs(shift) > 0.2:
                contrast_signal = {"previous_thought": self.last_thought, "valence_shift": round(shift, 2)}

        thought = self.thought_gen.generate_thought(
            self.state.focus,
            self.state.arousal,
            self.state.valence,
            prediction_errors[0] if prediction_errors else 0.0,
            last_five,
            self.self_model,
            curiosity,
            contrast_signal,
        )

        print("\n💭 Мысль агента:", thought)

        for s in stimuli:
            if s.get("type") == "interaction":
                reply = self.response_gen.generate(thought, s["content"], self.self_model)
                print("🗣 Ответ агента:", reply)

        print(f"  [Тренд валентности последних 5 событий: {trend_valence}]\n")
        self.last_thought = thought
        self.memory.save()  # 📤 сохраняем память

# ======================================================
# Simulation
# ======================================================

def run_interactive_simulation(steps: int = 100, timeout: int = 60):
    agent = Agent()
    print("💬 Пиши сообщение или жди — агент продолжает размышлять.")

    for _ in range(steps):
        try:
            user_input = inputimeout(prompt="Ты: ", timeout=timeout).strip()
        except TimeoutOccurred:
            user_input = ""

        stimuli = []

        if user_input:
            stimuli.append({
                "type": "interaction",
                "content": user_input,
                "salience": 0.9,
                "intensity": 0.4,
                "valence": 0.0,
            })
        else:
            stimuli.append({
                "type": "random",
                "content": random.choice(["мысль", "шум", "воспоминание"]),
                "salience": random.random(),
                "intensity": 0.3,
                "valence": random.uniform(-0.2, 0.2),
            })

        agent.step(stimuli)
        sleep(0.5)

if __name__ == "__main__":
    run_interactive_simulation()

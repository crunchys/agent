from dataclasses import dataclass
from time import time, sleep
from inputimeout import inputimeout, TimeoutOccurred
from typing import Optional, List, Dict
import random
from collections import Counter
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from agent_memory import PersistentMemory  # <=== подключаем память

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
# Meta Reflection
# ======================================================

class MetaReflection:
    def __init__(self, window: int = 5):
        self.window = window

    def reflect(self, memory: PersistentMemory) -> Optional[Dict]:
        recent = [e for e in memory.events if e["type"] == "experience"][-self.window:]
        if len(recent) < self.window:
            return None
        return {
            "type": "meta_reflection",
            "time": round(time(), 2),
            "avg_arousal": round(sum(e["arousal"] for e in recent) / self.window, 3),
            "avg_valence": round(sum(e["valence"] for e in recent) / self.window, 3),
        }

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
        recent = memory.recent(5)
        avg_valence = sum(e['valence'] for e in recent)/len(recent) if recent else 0
        self.traits['любопытство'] = min(1.0, max(0.0, 0.5 + avg_valence*0.5))

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
        surprise = 1.0 - exp['prob']
        return surprise

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
            "Ты — внутренняя мысль агента, возникающая сама по себе.\n"
            "Это не отчёт и не объяснение, а поток размышлений:\n"
            "— с сомнениями,\n"
            "— с оглядкой на прошлый опыт,\n"
            "— с попыткой понять, что происходит и почему.\n"
            "Ты можешь:\n"
            "— сравнивать текущее состояние с предыдущими,\n"
            "— колебаться в выводах,\n"
            "— признавать, что что-то пока неясно.\n"
            "Стиль — живая внутренняя речь, естественные фразы.\n"
            "Мысль должна быть законченной, без обрыва.\n"
            "Длина — 3–5 предложений.\n"
            "Не используй формулировки:\n"
            "«мне нужно», «я должен», «следует», «необходимо».\n"
            "В конце обязательно добавь <END_THOUGHT>."
        )

    def describe_affect(self, arousal: float, valence: float) -> str:
        if arousal < 0.2: arousal_desc = "спокойно"
        elif arousal < 0.5: arousal_desc = "настороженно"
        elif arousal < 0.8: arousal_desc = "активно"
        else: arousal_desc = "взволнованно"

        if valence < -0.5: valence_desc = "печально"
        elif valence < 0.0: valence_desc = "тревожно"
        elif valence < 0.5: valence_desc = "нейтрально"
        else: valence_desc = "радостно"

        return f"{arousal_desc}, {valence_desc}"

    def interpret_situation(self, focus, prediction_error, last_events):
        if not focus:
            return "происходящее пока не складывается в чёткую картину"
        if prediction_error > 0.6:
            return f"это ощущается как что-то неожиданное вокруг «{focus}»"
        if len(last_events) >= 2 and last_events[-1]["focus"] == last_events[-2]["focus"]:
            return f"кажется, «{focus}» повторяется и тянет за собой похожие состояния"
        return f"«{focus}» выглядит обыденно, но не совсем пусто"

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
            f"{e['focus']} (а:{e['arousal']:.2f}, v:{e['valence']:.2f})" for e in last_events
        )
        affect_desc = self.describe_affect(arousal, valence)

        contrast_text = ""
        if contrast_signal:
            contrast_text = "Ранее по этому поводу возникала другая мысль, и сейчас это ощущается как внутреннее расхождение.\n"

        situation_view = self.interpret_situation(focus, prediction_error, last_events)

        user_prompt = (
            f"Сейчас происходящее можно описать так: {situation_view}\n"
            f"Сейчас внимание удерживается на: {focus}\n"
            f"Общее состояние ощущается как: {affect_desc}\n"
            f"Любопытство на этом фоне — примерно {curiosity:.2f}\n"
            f"{contrast_text}"
            f"Ранее похожие состояния сопровождались такими событиями: {events_summary}\n"
            f"Черты личности сейчас проявляются так: {self_model.traits}\n"
            f"Есть ощущение, что ожидания не совсем совпали с происходящим (ошибка предсказания ≈ {prediction_error:.2f})"
        )

        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        output_ids = self.model.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=150,
            do_sample=True, temperature=1.0, top_p=0.95
        )
        generated = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if "<END_THOUGHT>" in text:
            text = text.split("<END_THOUGHT>")[0].strip()
        return text + " <END_THOUGHT>"

# ======================================================
# Response Generator (Speech)
# ======================================================

class ResponseGenerator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.system_prompt = (
            "Ты — ответ агента собеседнику.\n"
            "Ты осознаёшь, что говоришь с другим человеком.\n"
            "Ты не пересказываешь свои внутренние мысли напрямую.\n"
            "Ты отвечаешь естественно, по-человечески.\n"
            "Можно быть неуверенным, можно сомневаться.\n"
            "Ответ — 1–3 предложения."
        )

    def generate(self, thought: str, user_text: str, self_model):
        user_prompt = (
            f"Ты — {self_model.name}.\n"
            f"Собеседник сказал: «{user_text}».\n\n"
            f"Твоя внутренняя мысль в этот момент:\n{thought}\n\n"
            "Сформулируй ответ собеседнику."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )

        generated = output_ids[0][input_ids.shape[-1]:]
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
        self.memory.load()  # <=== подгружаем память и мир
        self.meta = MetaReflection()
        self.thought_gen = ThoughtGenerator()
        self.response_gen = ResponseGenerator(
            self.thought_gen.tokenizer,
            self.thought_gen.model
        )
        self.self_model = SelfModel()
        self.future = FutureExpectationSystem()
        self.last_thought: Optional[str] = None

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
            self.future.update(s['content'], s['valence'], s['intensity'])
            curiosity += self.future.curiosity(s['content'])

            # обновляем память о пользователе
            if s.get("type") == "interaction":
                self.memory.update_object(
                    "пользователь",
                    properties={"last_message": s["content"], "timestamp": time()}
                )

        curiosity /= len(stimuli) if stimuli else 1
        self.state.arousal = min(1.0, self.state.arousal + curiosity*0.2)

        self.state.focus = self.attention.select_focus(stimuli)
        self.state = self.emotion.decay(self.state)
        self.state.timestamp = time()

        self.memory.store({
            "type": "experience",
            "time": round(self.state.timestamp, 2),
            "focus": self.state.focus,
            "arousal": round(self.state.arousal, 3),
            "valence": round(self.state.valence, 3),
            "prediction_error": prediction_errors[0] if prediction_errors else 0.0,
        })

        # обновляем память о себе
        self.memory.update_object(
            "агент",
            properties={"last_focus": self.state.focus, "arousal": self.state.arousal, "valence": self.state.valence}
        )

        self.self_model.reflect(self.state, self.memory)
        last_five = self.memory.recent(5)
        trend_valence = "нейтральный"
        if len(last_five) >= 2:
            delta = last_five[-1]["valence"] - last_five[0]["valence"]
            trend_valence = "повышающийся" if delta > 0 else "падающий" if delta < 0 else "стабильный"

        contrast_signal = None
        if self.last_thought and len(last_five) >= 2:
            valence_shift = last_five[-1]["valence"] - last_five[-2]["valence"]
            if abs(valence_shift) > 0.2:
                contrast_signal = {"previous_thought": self.last_thought, "valence_shift": round(valence_shift, 2)}

        thought = self.thought_gen.generate_thought(
            self.state.focus, self.state.arousal, self.state.valence,
            prediction_errors[0] if prediction_errors else 0.0,
            last_five, self.self_model, curiosity,
            contrast_signal
        )

        print("\n💭 Мысль агента:", thought)

        # === Ответ собеседнику (если было обращение) ===
        for s in stimuli:
            if s.get("type") == "interaction":
                reply = self.response_gen.generate(
                    thought,
                    s["content"],
                    self.self_model
                )
                print("🗣 Ответ агента:", reply)

        print(f"  [Тренд валентности последних 5 событий: {trend_valence}]\n")
        self.last_thought = thought
        self.memory.save()  # <=== сохраняем память после каждого шага

# ======================================================
# Simulation
# ======================================================

def run_interactive_simulation(steps: int = 100, timeout: int = 60):
    agent = Agent()
    
    print("💬 Начинаем диалог с агентом. Пиши сообщение или просто жди — агент будет размышлять сам.")
    
    for _ in range(steps):
        try:
            user_input = inputimeout(prompt="Ты: ", timeout=timeout).strip()
        except TimeoutOccurred:
            user_input = ""  # если не ввел текст в течение timeout → пустой ввод
        
        stimuli = []

        if user_input:  # пользователь ввел текст → добавляем как стимул
            stimuli.append({
                "type": "interaction",
                "content": user_input,
                "salience": 0.9,
                "intensity": 0.4,
                "valence": 0.0,
            })
        else:  # пустой ввод или таймаут → генерируем случайное событие
            stimuli.append({
                "type": "random",
                "content": random.choice(["мысль", "шум", "воспоминание"]),
                "salience": random.random(),
                "intensity": 0.3,
                "valence": random.uniform(-0.2, 0.2),
            })
        
        agent.step(stimuli)
        sleep(0.5)  # небольшая пауза между шагами

if __name__ == "__main__":
    run_interactive_simulation()

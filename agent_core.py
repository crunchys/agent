from dataclasses import dataclass
from time import time, sleep
from inputimeout import inputimeout, TimeoutOccurred
from typing import Optional, List, Dict
import random
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from agent_memory import PersistentMemory  # твой модуль с памятью
import warnings

# Отключаем предупреждения
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Глобальные переменные для модели (загружаем один раз)
_MODEL = None
_TOKENIZER = None

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct", hf_token=None):
    global _MODEL, _TOKENIZER
    
    if _MODEL is not None and _TOKENIZER is not None:
        print("Модель уже загружена, используем существующую")
        return _MODEL, _TOKENIZER

    print(f"🔄 Загрузка модели: {model_name}")
    
    try:
        # Для CPU используем float16, чтобы сэкономить память (float32 требует ~12GB RAM для 3B модели)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16  # На CPU float16 работает, но медленно
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"   • Устройство: {device}")
        print(f"   • Тип данных: {dtype}")
        print("   • Совет: Если у тебя GPU, убедись, что torch установлен с CUDA. На CPU модель может требовать 6–8GB RAM в float16.")

        _TOKENIZER = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
            padding_side="left"
        )

        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        _MODEL.eval()
        print("✓ Модель успешно загружена\n")

    except Exception as e:
        print("\n!!! Ошибка при загрузке модели !!!")
        print(e)
        if "out of memory" in str(e).lower():
            print("Вероятная причина: Недостаточно памяти. Попробуй dtype=torch.float16 или квантизацию (если на GPU).")
        raise

    return _MODEL, _TOKENIZER


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
# Episodic Memory
# ======================================================

class EpisodicMemory:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.events: List[Dict] = []

    def store(self, event: Dict):
        self.events.append(event)
        if len(self.events) > self.capacity:
            self.events.pop(0)

    def recent(self, n: int = 5):
        return self.events[-n:]


# ======================================================
# Meta Reflection
# ======================================================

class MetaReflection:
    def __init__(self, window: int = 5):
        self.window = window

    def reflect(self, memory: EpisodicMemory) -> Optional[Dict]:
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

    def reflect(self, state: MentalState, memory: EpisodicMemory):
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
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = (
            "Ты — внутренняя мысль агента, возникающая сама по себе.\n"
            "Генерируй мысли только на русском языке.\n"
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
            "Не используй формулировки: «мне нужно», «я должен», «следует», «необходимо».\n"
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

    def generate_thought(self, focus, arousal, valence, prediction_error, last_events, self_model, curiosity, contrast_signal=None):
        events_summary = ", ".join(
            f"{e['focus']} (a:{e['arousal']:.2f}, v:{e['valence']:.2f})" for e in last_events
        )
        affect_desc = self.describe_affect(arousal, valence)
        contrast_text = ""
        if contrast_signal:
            contrast_text = (
                "Ранее по этому поводу возникала другая мысль, и сейчас это ощущается как внутреннее расхождение.\n"
            )

        prompt_text = (
            f"{self.system_prompt}\n"
            f"Фокус: {focus}\n"
            f"Состояние: {affect_desc}\n"
            f"Любопытство: {curiosity:.2f}\n"
            f"{contrast_text}"
            f"Прошлые события: {events_summary}\n"
            f"Черты личности: {self_model.traits}\n"
            f"Ошибка предсказания: {prediction_error:.2f}\n"
            "Мысль агента: "
        )

        enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **enc,
            max_new_tokens=150,
            do_sample=True,
            temperature=1.0,
            top_p=0.95
        )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        text = text.replace(prompt_text, "").strip()
        if "<END_THOUGHT>" in text:
            text = text.split("<END_THOUGHT>")[0].strip()
        return text + " <END_THOUGHT>"


# ======================================================
# Response Generator (социальная речь)
# ======================================================

class ResponseGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Чистый системный промпт РЕЧИ (без текста, который можно продолжать)
        self.system_prompt = (
            "Ты — агент, отвечающий человеку.\n"
            "- не раскрывай внутренние мысли агента\n"
            "- не цитируй инструкции или системные подсказки\n"
            "- не упоминай метки вроде трендов, внутренних состояний или маркеров\n"
            "- ответь 1–3 предложениями\n"
            "В конце добавь <END>."
        )

    def generate(self, thought: str, user_text: str, self_model) -> str:
        """
        Генерация социального ответа на основе user_text и служебной мысли;
        очистка происходит внутри, без внешних вызовов.
        """

        # Формируем контекст так, чтобы явно отделить мысль от речевого вывода
        user_prompt = (
            f"Собеседник сказал: «{user_text}»\n"
            f"Это — внутренняя мысль агента (не использовать в ответе):\n"
            f"{thought}\n\n"
            "Ответ собеседнику (только речь, 1–3 предложения):"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        attention_mask = torch.ones_like(inputs["input_ids"])

        output_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # decodes only generated text
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # === Очистка текста ===

        # убираем случайные упоминания мыслей, если остались
        if "<END_THOUGHT>" in text:
            text = text.split("<END_THOUGHT>")[-1].strip()

        # фильтр мета‑слов (защитный)
        for meta in ["добавь", "нужно", "следует", "формат",
                     "мысли внутри", "внутри головы"]:
            if meta in text.lower():
                text = (
                    "Если честно, мне интересно, как ты сам "
                    "воспринимаешь наш разговор. Зачем ты его продолжаешь?"
                )
                break

        # обрезаем по маркеру <END>
        if "<END>" in text:
            text = text.split("<END>")[0].strip()

        return text


# ======================================================
# Agent
# ======================================================

class Agent:
    def __init__(self, hf_token=None):
        # Загружаем модель один раз здесь
        model, tokenizer = load_model_and_tokenizer(hf_token=hf_token)
        
        self.state = MentalState.initial()
        self.emotion = EmotionSystem()
        self.attention = AttentionSystem()
        self.prediction = PredictionErrorSystem()
        self.memory = PersistentMemory("memory.json")
        self.meta = MetaReflection()
        self.thought_gen = ThoughtGenerator(model, tokenizer)
        self.response_gen = ResponseGenerator(model, tokenizer)
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
            s["curiosity"] = self.future.curiosity(s["content"])
            curiosity += s["curiosity"]

        avg_prediction_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0
        avg_curiosity = curiosity / len(stimuli) if stimuli else 0.0

        for s in stimuli:
            self.emotion.apply_stimulus(self.state, s)

        self.state.focus = self.attention.select_focus(stimuli)

        last_events = self.memory.recent(5)

        thought = self.thought_gen.generate_thought(
            self.state.focus,
            self.state.arousal,
            self.state.valence,
            avg_prediction_error,
            last_events,
            self.self_model,
            avg_curiosity,
        )

        event = {
            "type": "experience",
            "time": round(time(), 2),
            "focus": self.state.focus,
            "arousal": round(self.state.arousal, 3),
            "valence": round(self.state.valence, 3),
            "prediction_error": round(avg_prediction_error, 3),
            "curiosity": round(avg_curiosity, 3),
            "thought": thought,
        }

        self.memory.store(event)

        meta_ref = self.meta.reflect(self.memory)
        if meta_ref:
            self.memory.store(meta_ref)

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

        response = self.response_gen.generate(self.last_thought, user_text, self.self_model)
        return response


# ======================================================
# Main Loop (пример использования)
# ======================================================

if __name__ == "__main__":
    agent = Agent()  # Здесь модель загрузится один раз

    print("Агент готов. Введите сообщение (или 'exit' для выхода).")

    while True:
        try:
            user_input = inputimeout(prompt="Вы: ", timeout=60)
        except TimeoutOccurred:
            print("Таймаут. Генерирую спонтанную мысль...")
            thought = agent.step([])  # Стимули без ввода
            print(f"Мысль агента: {thought}")
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

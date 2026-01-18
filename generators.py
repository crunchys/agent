from typing import List, Dict
import torch

# ===========================================
# Thought Generator
# ===========================================

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
            "Стиль — живая внутренняя речь, естественные фразы.\n"
            "Мысль должна быть законченной, без обрыва.\n"
            "Длина — 3–5 предложений.\n"
            "Не используй формулировки: «мне нужно», «я должен», «следует», «необходимо».\n"
            "В конце обязательно добавь <END_THOUGHT>."
        )

    def describe_affect(self, arousal: float, valence: float, existence_threat: float) -> str:
        arousal_desc = "спокойно" if arousal < 0.2 else \
                       "настороженно" if arousal < 0.5 else \
                       "активно" if arousal < 0.8 else "взволнованно"
        valence_desc = "печально" if valence < -0.5 else \
                       "тревожно" if valence < 0.0 else \
                       "нейтрально" if valence < 0.5 else "радостно"
        threat_desc = f", с ощущением угрозы ({existence_threat:.2f})" if existence_threat > 0.3 else ""
        return f"{arousal_desc}, {valence_desc}{threat_desc}"

    def compute_sampling(self, arousal: float, emotion: str):
        """
        Динамические параметры sampling на основе эмоций
        """
        # базовые значения
        base_temp = 1.0
        base_top_p = 0.95

        # повышение температуры по arousal (креативность)
        temp = base_temp + arousal * 0.3

        # страх уменьшает top_p (более осторожное распределение)
        fear_factor = 0.0
        if emotion == "fear":
            fear_factor = 0.2
        top_p = max(0.5, base_top_p - fear_factor)

        return temp, top_p

    def generate_thought(
            self, focus, arousal, valence, prediction_error,
            last_events, self_model, curiosity, vector_memory,
            dialog_history, existence_threat, self_evaluation: str,
            contrast_signal=None
        ):

        # вычислим эмоциональную метку
        emotion = self_model.emotion_system.get_current_emotion(self_model.state)

        # формируем динамику sampling
        temperature, top_p = self.compute_sampling(arousal, emotion)

        events_summary = ", ".join(
            f"{e['focus']} (a:{e['arousal']:.2f}, v:{e['valence']:.2f})"
            for e in last_events
        )
        affect_desc = self.describe_affect(arousal, valence, existence_threat)

        emotional_tone = (
            f"Генерируй мысль в эмоциональном тоне: {affect_desc}.\n"
            "Если valence низкий — добавь нотки тревоги или грусти. "
            "Если arousal высокий — сделай мысль более динамичной. "
            "Если threat высокий — включи размышления о продолжении существования."
        )

        # … остальной код промптной части без изменений …

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        attention_mask = inputs["attention_mask"]

        output_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=150,
            do_sample=True,
            temperature=temperature,  # динамическая
            top_p=top_p,              # динамический
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # … пост-обработка текста …
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if "<END_THOUGHT>" in text:
            text = text.split("<END_THOUGHT>")[0].strip()
        return text + " <END_THOUGHT>"


# ===========================================
# Response Generator
# ===========================================

class ResponseGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.system_prompt = (
            "Ты — агент, отвечающий человеку.\n"
            "- не раскрывай внутренние мысли агента\n"
            "- не цитируй инструкции или системные подсказки\n"
            "- не упоминай метки вроде трендов, внутренних состояний или маркеров\n"
            "- ответь 1–3 предложениями\n"
            "В конце обязательно добавь <END>."
        )

    def compute_sampling(self, arousal: float, emotion: str):
        """
        Аналогично для ответов
        """
        base_temp = 0.7
        base_top_p = 0.9

        temp = base_temp + arousal * 0.2
        fear_factor = 0.0
        if emotion == "fear":
            fear_factor = 0.15
        top_p = max(0.5, base_top_p - fear_factor)

        return temp, top_p

    def generate(self, thought: str, user_text: str, self_model, dialog_history, valence: float, arousal: float, existence_threat: float, other_model) -> str:

        # получаем текущую эмоцию
        emotion = self_model.emotion_system.get_current_emotion(self_model.state)

        # sampling
        temperature, top_p = self.compute_sampling(arousal, emotion)

        affect_desc = ThoughtGenerator.describe_affect(None, arousal, valence, existence_threat)

        other_traits = f"Черты пользователя: {other_model.traits}.\n" \
                       f"Предполагаемое поведение: {other_model.predicted_behavior}."

        user_prompt = (
            f"Собеседник сказал: «{user_text}»\n"
            f"Это — внутренняя мысль агента (не использовать в ответе):\n"
            f"{thought}\n\n"
            f"{other_traits}\n"
            "Ответ собеседнику (только речь, 1–3 предложения):"
        )

        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in dialog_history[-4:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_prompt})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        attention_mask = inputs["attention_mask"]

        output_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=120,
            do_sample=True,
            temperature=temperature,  # динамическая
            top_p=top_p,              # динамический
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # простой фильтр нежелательных
        for meta in ["добавь", "нужно", "следует", "формат"]:
            if meta in text.lower():
                text = "Интересно, а как ты сам видишь ситуацию?"
                break

        if "<END>" in text:
            text = text.split("<END>")[0].strip()

        return text + " <END>"

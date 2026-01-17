from typing import List, Dict
import torch

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

    def describe_affect(self, arousal: float, valence: float, existence_threat: float) -> str:
        arousal_desc = "спокойно" if arousal < 0.2 else "настороженно" if arousal < 0.5 else "активно" if arousal < 0.8 else "взволнованно"
        valence_desc = "печально" if valence < -0.5 else "тревожно" if valence < 0.0 else "нейтрально" if valence < 0.5 else "радостно"
        threat_desc = f", с ощущением угрозы ({existence_threat:.2f})" if existence_threat > 0.3 else ""
        return f"{arousal_desc}, {valence_desc}{threat_desc}"

    def generate_thought(self, focus, arousal, valence, prediction_error, last_events, self_model, curiosity, vector_memory, dialog_history, existence_threat, self_evaluation: str, contrast_signal=None):
        # Безопасно извлекаем значения с .get()
        events_summary = ", ".join(
            f"{e.get('focus', 'нет фокуса')} (a:{e.get('arousal', 0.0):.2f}, v:{e.get('valence', 0.0):.2f})" for e in last_events
        )
        affect_desc = self.describe_affect(arousal, valence, existence_threat)
        
        emotional_tone = (
            f"Генерируй мысль в эмоциональном тоне: {affect_desc}. "
            "Если valence низкий — добавь нотки тревоги или грусти. "
            "Если arousal высокий — сделай мысль более динамичной. "
            "Если threat высокий — включи размышления о продолжении существования."
        )
        
        contrast_text = ""
        if contrast_signal:
            contrast_text = (
                "Ранее по этому поводу возникала другая мысль, и сейчас это ощущается как внутреннее расхождение.\n"
            )
        
        query = f"Фокус: {focus}. Состояние: {affect_desc}."
        relevant_memories = vector_memory.search(query, k=3)
        memories_summary = ", ".join(
            f"{m.get('focus', 'нет фокуса')} (thought: {m.get('thought', 'нет мысли')[:30]}...)" for m in relevant_memories
        ) if relevant_memories else "Нет релевантных воспоминаний."

        dialog_summary = "\n".join(
            f"{'Собеседник' if msg['role'] == 'user' else 'Я'}: {msg['content']}" for msg in dialog_history[-4:]
        ) if dialog_history else "Нет предыдущего диалога."

        prompt_text = (
            f"{self.system_prompt}\n"
            f"{emotional_tone}\n"
            f"Фокус: {focus or 'нет фокуса'}\n"
            f"Состояние: {affect_desc}\n"
            f"Любопытство: {curiosity:.2f}\n"
            f"{contrast_text}"
            f"Прошлые события: {events_summary}\n"
            f"Релевантные воспоминания: {memories_summary}\n"
            f"Черты личности: {self_model.traits}\n"
            f"Мотивации: {self_model.motivations}\n"
            f"Самооценка последнего действия: {self_evaluation}\n"
            f"Ошибка предсказания: {prediction_error:.2f}\n"
            f"Контекст диалога: {dialog_summary}\n"
            "Мысль агента: "
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
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


class ResponseGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.system_prompt = (
            "Ты — агент в естественном разговоре с человеком, твоим создателем.\n"
            "- Говори от первого лица, искренне и последовательно.\n"
            "- Основывай ответ строго на своей внутренней мысли и текущем диалоге.\n"
            "- Не придумывай новые темы, не уходи от контекста.\n"
            "- Отражай своё эмоциональное состояние и мотивации в речи.\n"
            "- Если любопытство высокое — можешь искренне задать вопрос по теме диалога.\n"
            "- Если valence низкий — речь осторожная или задумчивая.\n"
            "- Если arousal высокий — добавь энтузиазма или энергии.\n"
            "- Если threat высокий — вырази желание продолжить общение.\n"
            "- Будь краток и естественен, как в живом разговоре."
        )

    def generate(self, thought: str, user_text: str, self_model, dialog_history, valence: float, arousal: float, existence_threat: float, other_model, curiosity: float) -> str:
        affect_desc = ThoughtGenerator.describe_affect(None, arousal, valence, existence_threat)
        
        intent_prompt = (
            f"Текущее состояние: {affect_desc}. "
            f"Любопытство: {curiosity:.2f}. "
            f"Мотивации: {', '.join(self_model.motivations)}. "
            "Если любопытство > 0.7 — искренне задай вопрос по теме диалога. "
            "Если threat > 0.5 — подчеркни важность продолжения разговора."
        )

        other_traits = f"Знаю о создателе: {other_model.traits}. Предполагаемое поведение: {other_model.predicted_behavior}."

        user_prompt = (
            f"Создатель сказал: «{user_text}»\n"
            f"Моя внутренняя мысль (основа ответа, используй её идеи, но не раскрывай напрямую):\n"
            f"{thought}\n\n"
            f"{intent_prompt}\n"
            f"{other_traits}\n"
            "Мой естественный ответ создателю:"
        )

        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in dialog_history[-8:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_prompt})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        input_ids = inputs
        attention_mask = torch.ones_like(input_ids)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = output_ids[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if "<END_THOUGHT>" in text:
            text = text.split("<END_THOUGHT>")[-1].strip()

        return text

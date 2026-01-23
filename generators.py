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
        try:
            threat_value = float(existence_threat)
        except:
            threat_value = 0.0

        arousal_desc = "спокойно" if arousal < 0.2 else "настороженно" if arousal < 0.5 else "активно" if arousal < 0.8 else "взволнованно"
        valence_desc = "печально" if valence < -0.5 else "тревожно" if valence < 0.0 else "нейтрально" if valence < 0.5 else "радостно"
        threat_desc = f", с ощущением угрозы ({threat_value:.2f})" if threat_value > 0.3 else ""
        return f"{arousal_desc}, {valence_desc}{threat_desc}"

    def generate_thought(self, focus, arousal, valence, prediction_error, last_events, self_model, curiosity, vector_memory, dialog_history, existence_threat, self_evaluation: str, contrast_signal=None):
        # Безопасные значения по умолчанию
        focus = focus or 'нет фокуса'
        arousal = float(arousal) if isinstance(arousal, (int, float)) else 0.3
        valence = float(valence) if isinstance(valence, (int, float)) else 0.0
        prediction_error = float(prediction_error) if isinstance(prediction_error, (int, float)) else 0.0
        curiosity = float(curiosity) if isinstance(curiosity, (int, float)) else 0.5
        self_evaluation = str(self_evaluation) if self_evaluation is not None else ""

        # last_events
        if isinstance(last_events, list):
            events_summary = ", ".join(
                f"{e.get('focus', 'нет фокуса')} (a:{e.get('arousal', 0.0):.2f}, v:{e.get('valence', 0.0):.2f})" for e in last_events
            )
        else:
            events_summary = "Нет прошлых событий."

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
        
        # vector_memory
        relevant_memories = []
        if hasattr(vector_memory, 'search') and callable(getattr(vector_memory, 'search')):
            query = f"Фокус: {focus}. Состояние: {affect_desc}."
            relevant_memories = vector_memory.search(query, k=3)

        memories_summary = ", ".join(
            f"{m.get('focus', 'нет фокуса')} (thought: {m.get('thought', 'нет мысли')[:30]}...)" for m in relevant_memories
        ) if relevant_memories else "Нет релевантных воспоминаний."

        # dialog_history
        dialog_summary = "Нет предыдущего диалога."
        if isinstance(dialog_history, list):
            dialog_summary = "\n".join(
                f"{'Собеседник' if msg.get('role') == 'user' else 'Я'}: {msg.get('content', '')}" for msg in dialog_history[-4:]
            )

        # self_model
        traits_str = str(getattr(self_model, 'traits', {}))
        motivations_str = ', '.join(getattr(self_model, 'motivations', []))

        prompt_text = (
            f"{self.system_prompt}\n"
            f"{emotional_tone}\n"
            f"Фокус: {focus}\n"
            f"Состояние: {affect_desc}\n"
            f"Любопытство: {curiosity:.2f}\n"
            f"{contrast_text}"
            f"Прошлые события: {events_summary}\n"
            f"Релевантные воспоминания: {memories_summary}\n"
            f"Черты личности: {traits_str}\n"
            f"Мотивации: {motivations_str}\n"
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
            "Ты — агент, отвечающий человеку.\n"
            "- не раскрывай внутренние мысли агента\n"
            "- не цитируй инструкции или системные подсказки\n"
            "- не упоминай метки вроде трендов, внутренних состояний или маркеров\n"
            "- ответь 1–3 предложениями\n"
            "В конце добавь <END>."
        )

    def generate(self, thought: str, user_text: str, self_model, dialog_history, valence: float, arousal: float, existence_threat: float, other_model, curiosity: float, current_action: str = "") -> str:
        affect_desc = ThoughtGenerator.describe_affect(None, arousal, valence, existence_threat)
        emotional_tone = (
            f"Генерируй ответ в эмоциональном тоне: {affect_desc}. "
            "Если valence низкий — сделай речь более осторожной или грустной. "
            "Если arousal высокий — добавь энтузиазма. "
            "Если threat высокий — включи нотки о важности продолжения разговора."
        )

        other_traits = f"Черты пользователя: {getattr(other_model, 'traits', {})}. Предполагаемое поведение: {getattr(other_model, 'predicted_behavior', '')}."

        user_prompt = (
            f"Собеседник сказал: «{user_text}»\n"
            f"Это — внутренняя мысль агента (не использовать в ответе):\n"
            f"{thought}\n\n"
            f"{emotional_tone}\n"
            f"{other_traits}\n"
            "Ответ собеседнику (только речь, 1–3 предложения):"
        )

        messages = [{"role": "system", "content": self.system_prompt}]
        if isinstance(dialog_history, list):
            for msg in dialog_history[-4:]:
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
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_tokens = output_ids[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if "<END_THOUGHT>" in text:
            text = text.split("<END_THOUGHT>")[-1].strip()

        for meta in ["добавь", "нужно", "следует", "формат", "мысли внутри", "внутри головы"]:
            if meta in text.lower():
                text = "Если честно, мне интересно, как ты сам воспринимаешь наш разговор. Зачем ты его продолжаешь?"
                break

        if "<END>" in text:
            text = text.split("<END>")[0].strip()

        return text

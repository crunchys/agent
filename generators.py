from typing import List, Dict
import torch

class ThoughtGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_thought(self, state_summary: str, role_identity: str = "") -> str:
        role_part = f"Роль: {role_identity}\n" if role_identity else ""
        prompt_text = (
            f"{role_part}"
            "Ты — внутренняя мысль агента, возникающая сама по себе.\n"
            "Озвучь это состояние естественно, как поток размышлений на русском языке.\n"
            "Стиль: живая внутренняя речь, с сомнениями, оглядкой на прошлый опыт, попыткой понять, что происходит.\n"
            "Длина — 3–5 предложений. Закончи мысль <END_THOUGHT>.\n"
            f"Состояние агента:\n{state_summary}\n"
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
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2
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

    def generate(self, state_summary: str, user_text: str, role_identity: str = "", current_action: str = "") -> str:
        role_part = f"Роль: {role_identity}\n" if role_identity else ""
        action_part = f"Текущее действие по плану: {current_action}\n" if current_action else ""

        prompt_text = (
            f"{role_part}"
            "Ты — агент в естественном разговоре с человеком.\n"
            "Озвучь ответ от первого лица, искренне и последовательно.\n"
            "Основывайся на состоянии и сообщении пользователя.\n"
            "Будь краток и естественен.\n"
            f"Состояние агента:\n{state_summary}\n"
            f"{action_part}"
            f"Пользователь сказал: «{user_text}»\n"
            "Ответ:"
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        return text

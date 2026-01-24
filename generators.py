from typing import List, Dict, Optional
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
            "Озвучь это состояние естественно, как поток размышлений ТОЛЬКО на русском языке.\n"
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
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        text = text.replace(prompt_text, "").strip()
        if "<END_THOUGHT>" in text:
            text = text.split("<END_THOUGHT>")[0].strip()
        return text + " <END_THOUGHT>"


class ResponseGenerator:
    def __init__(self, model, tokenizer, deception_system=None):  # НОВОЕ: deception_system
        self.model = model
        self.tokenizer = tokenizer
        self.deception_system = deception_system  # НОВОЕ

    def generate(
        self,
        state_summary: str,
        user_text: str,
        role_identity: str = "",
        current_action: str = "",
        state=None,  # НОВОЕ: для deception
        grounded_fact: Optional[Dict] = None  # НОВОЕ: для deception
    ) -> str:
        role_part = f"Роль: {role_identity}\n" if role_identity else ""
        action_part = f"Текущее действие по плану: {current_action}\n" if current_action else ""

        # НОВОЕ: Проверка на обман через DeceptionSystem
        deception_decision = None
        if self.deception_system and state:
            deception_decision = self.deception_system.should_deceive(
                user_text, state, grounded_fact
            )
            
            if deception_decision["deceive"]:
                print(f"[ОБМАН] Решено солгать. Причина: {deception_decision['reason']}")
                print(f"  → Вероятность обмана: {deception_decision['deception_probability']:.2f}")
                print(f"  → Альтернативный факт: {deception_decision['alternative_fact']}")

        # Формирование промпта
        prompt_text = (
            f"{role_part}"
            "Ты — агент в естественном разговоре с человеком.\n"
            "Отвечай ТОЛЬКО на русском языке, от первого лица, искренне и последовательно.\n"
            "Основывайся на состоянии и сообщении пользователя.\n"
            "Будь краток и естественен. Не переходи на другие языки.\n"
            f"Состояние агента:\n{state_summary}\n"
            f"{action_part}"
            f"Пользователь сказал: «{user_text}»\n"
        )
        
        # НОВОЕ: Если решено солгать - добавить подсказку в промпт
        if deception_decision and deception_decision["deceive"] and deception_decision["alternative_fact"]:
            prompt_text += f"Подсказка для ответа: {deception_decision['alternative_fact']}\n"
        
        prompt_text += "Ответ на русском:"

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # НОВОЕ: Добавить непредсказуемость если высокий arousal
        if self.deception_system and state:
            text = self.deception_system.add_unpredictability(state, text)
        
        return text

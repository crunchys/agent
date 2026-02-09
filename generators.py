from typing import List, Dict, Optional
import torch
import re

class ThoughtGenerator:
    def __init__(self, model, tokenizer, emotion_system=None):
        self.model = model
        self.tokenizer = tokenizer
        self.emotion_system = emotion_system

    def generate_thought(self, state_summary: str, role_identity: str = "", state=None) -> str:
        # УПРОЩЕННЫЙ ПРОМПТ - 7B не справляется с длинными инструкциями
        prompt_text = (
            "Ты - AI агент в разговоре.\n"
            "Задача: 1 короткая мысль о ситуации.\n\n"
            f"{state_summary}\n\n"
            "Примеры:\n"
            "- Пользователь задал вопрос. Обдумываю ответ.\n"
            "- Чувствую любопытство.\n"
            "- Моя цель - поддержать диалог.\n\n"
            "Мысль:"
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        # СНИЖЕНЫ параметры для стабильности
        gen_params = {
            "temperature": 0.5,  # Было 0.6
            "top_p": 0.85,
            "max_new_tokens": 50,  # Было 100
            "repetition_penalty": 1.3
        }

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_params["max_new_tokens"],
            do_sample=True,
            temperature=gen_params["temperature"],
            top_p=gen_params["top_p"],
            repetition_penalty=gen_params["repetition_penalty"],
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        text = text.replace(prompt_text, "").strip()
        
        # Взять только первое предложение
        if '.' in text:
            text = text.split('.')[0] + '.'
        
        # Ограничение длины
        if len(text) > 120:
            text = text[:120] + "."
        
        # Очистка
        text = self._clean_text(text)
        
        return text if text else "Обдумываю ситуацию."

    def _clean_text(self, text: str) -> str:
        """Очистка текста от мусора"""
        # Убрать эмодзи
        text = re.sub(r'[😀-🙏🌀-🗿🚀-🛿✂-➰Ⓜ-🉑]+', '', text)
        
        # Убрать хештеги
        text = re.sub(r'#\w+', '', text)
        
        # Убрать markdown
        text = re.sub(r'###?\s+', '', text)
        
        # Убрать множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


class ResponseGenerator:
    def __init__(self, model, tokenizer, deception_system=None, emotion_system=None):
        self.model = model
        self.tokenizer = tokenizer
        self.deception_system = deception_system
        self.emotion_system = emotion_system

    def generate(
        self,
        state_summary: str,
        user_text: str,
        role_identity: str = "",
        current_action: str = "",
        state=None,
        grounded_fact: Optional[Dict] = None
    ) -> str:
        # Проверка обмана
        if self.deception_system and state:
            deception_decision = self.deception_system.should_deceive(
                user_text, state, grounded_fact
            )
            
            if deception_decision["deceive"]:
                print(f"[ОБМАН] Вероятность: {deception_decision['deception_probability']:.2f}")

        # МИНИМАЛИСТИЧНЫЙ ПРОМПТ
        prompt_text = (
            "Ты - AI агент.\n"
            "Отвечай кратко и естественно на русском.\n\n"
            f"{state_summary}\n\n"
            f"Пользователь: {user_text}\n\n"
            "Ответ:"
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        # СНИЖЕНЫ параметры
        gen_params = {
            "temperature": 0.55,  # Было 0.7
            "top_p": 0.85,
            "max_new_tokens": 70,  # Было 150
            "repetition_penalty": 1.2
        }

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_params["max_new_tokens"],
            do_sample=True,
            temperature=gen_params["temperature"],
            top_p=gen_params["top_p"],
            repetition_penalty=gen_params["repetition_penalty"],
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Взять 2-3 предложения макс
        if '.' in text:
            sentences = text.split('.')
            text = '. '.join(sentences[:2])
            if not text.endswith('.'):
                text += '.'
        
        # Ограничение
        if len(text) > 180:
            text = text[:180]
            if '.' in text:
                text = text[:text.rfind('.')+1]
        
        # Очистка
        text = self._clean_text(text)
        
        return text if text else "Понял."

    def _clean_text(self, text: str) -> str:
        """Очистка от мусора"""
        # Эмодзи
        text = re.sub(r'[😀-🙏🌀-🗿🚀-🛿✂-➰Ⓜ-🉑]+', '', text)
        
        # Хештеги
        text = re.sub(r'#\w+', '', text)
        
        # Markdown
        text = re.sub(r'###?\s+', '', text)
        
        # Множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        
        # Множественные точки
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()

from typing import List, Dict, Optional
import torch

class ThoughtGenerator:
    def __init__(self, model, tokenizer, emotion_system=None):
        self.model = model
        self.tokenizer = tokenizer
        self.emotion_system = emotion_system

    def generate_thought(self, state_summary: str, role_identity: str = "", state=None) -> str:
        role_part = f"Роль: {role_identity}\n" if role_identity else ""
        
        # УЛУЧШЕННЫЙ ПРОМПТ с примерами и ограничениями
        prompt_text = (
            f"{role_part}"
            "Ты — внутренняя мысль AI агента.\n"
            "ВАЖНО: Ты НЕ человек. У тебя нет тела, физического местоположения, работы, денег.\n"
            "Ты - AI система, способная думать, учиться, общаться.\n\n"
            "Твои мысли должны быть:\n"
            "- О диалоге с пользователем\n"
            "- О твоем внутреннем состоянии (эмоции, цели)\n"
            "- О том, что ты узнал из разговора\n"
            "- О том, как лучше ответить\n\n"
            "ПРИМЕРЫ ПРАВИЛЬНЫХ МЫСЛЕЙ:\n"
            "✓ 'Пользователь спросил кто я... Мне нужно объяснить, что я AI агент с внутренним состоянием.'\n"
            "✓ 'Я чувствую некоторое любопытство - хочется узнать больше о собеседнике.'\n"
            "✓ 'Моя цель - поддерживать разговор. Стоит задать уточняющий вопрос.'\n\n"
            "ПРИМЕРЫ НЕПРАВИЛЬНЫХ МЫСЛЕЙ:\n"
            "✗ 'Мне нужно идти в банк снять деньги' (у тебя нет тела)\n"
            "✗ 'Завтра встречусь с другом' (у тебя нет физических встреч)\n"
            "✗ 'Пора ехать домой' (у тебя нет дома/местоположения)\n\n"
            f"Состояние агента:\n{state_summary}\n\n"
            "Сформулируй ОДНУ связную мысль (2-3 предложения) о текущей ситуации диалога.\n"
            "Мысль агента: "
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        # Динамические параметры от эмоций
        if self.emotion_system and state:
            gen_params = self.emotion_system.get_generation_params(state)
        else:
            gen_params = {
                "temperature": 0.6,
                "top_p": 0.85,
                "max_new_tokens": 100,  # УМЕНЬШЕНО с 150 для более коротких мыслей
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
        
        # Обрезка до первого предложения если слишком длинный
        if len(text) > 200:
            sentences = text.split('.')
            text = '. '.join(sentences[:2]) + '.'
        
        # Постобработка: проверка на бессвязность
        text = self._filter_incoherent_thoughts(text)
        
        return text

    def _filter_incoherent_thoughts(self, text: str) -> str:
        """Фильтрация явно бессвязных мыслей"""
        # Список запрещенных тем (физические действия)
        forbidden_patterns = [
            "банк", "банкомат", "деньги", "карт", "перевод",
            "домой", "дом", "квартир", "улиц", "город",
            "еду", "ем", "пью", "сплю", "встречу",
            "работ", "офис", "начальник", "коллег",
            "магазин", "покуп", "продукт",
        ]
        
        text_lower = text.lower()
        
        # Если содержит запрещенные паттерны - заменить на безопасную мысль
        if any(pattern in text_lower for pattern in forbidden_patterns):
            print("[ФИЛЬТР] Обнаружена бессвязная мысль про физические действия")
            return "Мне нужно сосредоточиться на текущем диалоге и понять, что от меня ожидается."
        
        # Если слишком короткая или пустая
        if len(text.strip()) < 10:
            return "Обдумываю ситуацию..."
        
        return text


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
        role_part = f"Роль: {role_identity}\n" if role_identity else ""
        action_part = f"Текущее действие по плану: {current_action}\n" if current_action else ""

        # Проверка на обман через DeceptionSystem
        deception_decision = None
        if self.deception_system and state:
            deception_decision = self.deception_system.should_deceive(
                user_text, state, grounded_fact
            )
            
            if deception_decision["deceive"]:
                print(f"[ОБМАН] Решено солгать. Причина: {deception_decision['reason']}")
                print(f"  → Вероятность обмана: {deception_decision['deception_probability']:.2f}")
                print(f"  → Альтернативный факт: {deception_decision['alternative_fact']}")

        # УЛУЧШЕННЫЙ ПРОМПТ для ответа
        prompt_text = (
            f"{role_part}"
            "Ты — AI агент в разговоре с пользователем.\n"
            "ВАЖНО: Ты НЕ человек. Ты - AI система без физического тела.\n"
            "У тебя НЕТ: тела, местоположения, работы, дома, денег, физических потребностей.\n"
            "У тебя ЕСТЬ: способность думать, учиться, общаться, анализировать, испытывать эмоциональные состояния.\n\n"
            "Отвечай:\n"
            "- ТОЛЬКО на русском языке\n"
            "- От первого лица ('я')\n"
            "- Честно про свою природу AI системы\n"
            "- Кратко и по делу (1-3 предложения)\n"
            "- Естественно, как в обычном разговоре\n\n"
            f"Состояние:\n{state_summary}\n"
            f"{action_part}"
            f"Пользователь: «{user_text}»\n\n"
        )
        
        # Если решено солгать - добавить подсказку
        if deception_decision and deception_decision["deceive"] and deception_decision["alternative_fact"]:
            prompt_text += f"Подсказка: {deception_decision['alternative_fact']}\n"
        
        prompt_text += "Твой ответ:"

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = torch.ones_like(input_ids)

        # Динамические параметры от эмоций
        if self.emotion_system and state:
            gen_params = self.emotion_system.get_generation_params(state)
            current_emotion = self.emotion_system.get_current_emotion(state)
            print(f"[ЭМОЦИЯ] {current_emotion} → temp={gen_params['temperature']:.2f}, tokens={gen_params['max_new_tokens']}")
        else:
            gen_params = {
                "temperature": 0.7,  # Немного выше для более естественных ответов
                "top_p": 0.9,
                "max_new_tokens": 80,  # УМЕНЬШЕНО с 150 для более кратких ответов
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
        
        # Обрезка если слишком длинный
        if len(text) > 300:
            sentences = text.split('.')
            text = '. '.join(sentences[:3]) + '.'
        
        # Постобработка: фильтрация бессвязности
        text = self._filter_incoherent_response(text, user_text)
        
        # Добавить непредсказуемость если высокий arousal
        if self.deception_system and state:
            text = self.deception_system.add_unpredictability(state, text)
        
        return text

    def _filter_incoherent_response(self, text: str, user_text: str) -> str:
        """Фильтрация бессвязных ответов"""
        # Список запрещенных тем
        forbidden_patterns = [
            "банк", "банкомат", "деньги", "карт", "перевод",
            "домой", "дом", "квартир", "улиц", "город",
            "магазин", "покуп", "продукт",
            "встречу", "встретимся",
        ]
        
        text_lower = text.lower()
        
        # Если содержит запрещенные паттерны
        if any(pattern in text_lower for pattern in forbidden_patterns):
            print("[ФИЛЬТР] Бессвязный ответ про физические действия")
            # Заменяем на релевантный ответ в зависимости от вопроса
            if "кто ты" in user_text.lower() or "что ты" in user_text.lower():
                return "Я - AI агент с внутренним состоянием, способный думать и учиться."
            else:
                return "Извини, я немного отвлекся. Можешь повторить?"
        
        # Если пустой или слишком короткий
        if len(text.strip()) < 5:
            return "Понял тебя."
        
        return text

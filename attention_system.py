from typing import List, Dict, Optional
from time import time

class AttentionSystem:
    """
    Улучшенная система внимания.
    
    Включает:
    - Bottom-up attention (salience-based)
    - Top-down attention (goal-driven)
    - Attention switching (переключение фокуса)
    - Attention span (продолжительность фокуса)
    - Focus stability (стабильность внимания)
    """
    
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        
        # НОВОЕ: Отслеживание текущего фокуса
        self.current_focus = None
        self.focus_start_time = None
        self.focus_duration = 0.0
        
        # НОВОЕ: История фокусов
        self.focus_history = []  # [(focus, duration, reason), ...]
        
        # НОВОЕ: Параметры attention span
        self.min_focus_duration = 2.0  # Минимум 2 секунды на фокус
        self.max_focus_duration = 30.0  # Максимум 30 секунд
        
        # НОВОЕ: Switching cost (стоимость переключения)
        self.switching_cost = 0.2  # Новый стимул должен быть на 20% важнее
    
    def select_focus(
        self, 
        stimuli: List[Dict],
        current_goal: Optional[str] = None,
        force_switch: bool = False
    ) -> Optional[str]:
        """
        Выбрать фокус внимания.
        
        Args:
            stimuli: Список стимулов с salience
            current_goal: Текущая цель агента (для top-down)
            force_switch: Принудительно переключить (игнорировать span)
        
        Returns:
            Содержимое фокуса или None
        """
        if not stimuli:
            return self._maintain_or_clear_focus()
        
        # 1. Bottom-up attention (от стимулов)
        bottom_up_focus = self._select_by_salience(stimuli)
        
        # 2. Top-down attention (от цели)
        top_down_focus = self._select_by_goal(stimuli, current_goal)
        
        # 3. Объединение bottom-up и top-down
        candidate_focus = self._combine_attention(
            bottom_up_focus, 
            top_down_focus,
            stimuli
        )
        
        # 4. Проверка switching (нужно ли переключаться?)
        should_switch = self._should_switch(
            candidate_focus,
            force_switch
        )
        
        if should_switch:
            # Переключаемся на новый фокус
            return self._switch_focus(candidate_focus, "new_stimulus")
        else:
            # Удерживаем текущий фокус
            return self._maintain_focus()
    
    def _select_by_salience(self, stimuli: List[Dict]) -> Optional[Dict]:
        """Bottom-up: выбор по важности (salience)"""
        if not stimuli:
            return None
        
        # Находим самый важный стимул
        best = max(stimuli, key=lambda s: s.get("salience", 0))
        
        # Проверяем threshold
        if best.get("salience", 0) >= self.threshold:
            return best
        
        return None
    
    def _select_by_goal(
        self, 
        stimuli: List[Dict], 
        current_goal: Optional[str]
    ) -> Optional[Dict]:
        """Top-down: выбор по соответствию цели"""
        if not current_goal:
            return None
        
        # НОВОЕ: Поиск стимула релевантного цели
        goal_lower = current_goal.lower()
        
        for stimulus in stimuli:
            content = stimulus.get("content", "").lower()
            
            # Простая эвристика: есть ли ключевые слова цели в стимуле?
            # Примеры целей:
            # - "Узнать больше о собеседнике" → ищем вопросы от пользователя
            # - "Поддерживать разговор" → ищем активность пользователя
            # - "Разобраться в X" → ищем упоминания X
            
            if "узнать" in goal_lower or "вопрос" in goal_lower:
                # Цель = задать вопрос → фокус на последний ввод пользователя
                if "content" in stimulus:
                    return stimulus
            
            if "поддержать" in goal_lower or "разговор" in goal_lower:
                # Цель = поддержать разговор → фокус на активность
                if stimulus.get("salience", 0) > 0.3:
                    return stimulus
            
            if "разобраться" in goal_lower:
                # Цель = разобраться → фокус на непонятное (prediction_error)
                if stimulus.get("prediction_error", 0) > 0.5:
                    return stimulus
        
        return None
    
    def _combine_attention(
        self,
        bottom_up: Optional[Dict],
        top_down: Optional[Dict],
        stimuli: List[Dict]
    ) -> Optional[str]:
        """
        Объединение bottom-up и top-down внимания.
        
        Правило:
        - Если есть top-down (от цели) → приоритет ему (вес 0.7)
        - Иначе bottom-up (от важности) (вес 0.3)
        """
        if top_down and bottom_up:
            # Оба есть - взвешенный выбор
            top_down_score = (top_down.get("salience", 0) * 0.7 + 
                             top_down.get("prediction_error", 0) * 0.3)
            bottom_up_score = bottom_up.get("salience", 0)
            
            if top_down_score >= bottom_up_score:
                return top_down.get("content")
            else:
                return bottom_up.get("content")
        
        elif top_down:
            return top_down.get("content")
        
        elif bottom_up:
            return bottom_up.get("content")
        
        return None
    
    def _should_switch(
        self, 
        candidate_focus: Optional[str],
        force_switch: bool
    ) -> bool:
        """
        Решить, нужно ли переключить фокус.
        
        Правила:
        1. Если force_switch=True → переключаем
        2. Если нет текущего фокуса → переключаем
        3. Если фокус слишком долго (> max_duration) → переключаем
        4. Если фокус слишком короткий (< min_duration) → НЕ переключаем (стабильность)
        5. Если новый кандидат НАМНОГО важнее → переключаем (учитывая switching_cost)
        """
        # Правило 1: Принудительное переключение
        if force_switch:
            return True
        
        # Правило 2: Нет текущего фокуса
        if self.current_focus is None:
            return True
        
        # Правило 3: Слишком долго на одном фокусе
        if self.focus_start_time:
            current_duration = time() - self.focus_start_time
            if current_duration > self.max_focus_duration:
                print(f"[ATTENTION] Переключение: фокус слишком долго ({current_duration:.1f}s)")
                return True
        
        # Правило 4: Слишком короткий фокус (стабильность)
        if self.focus_start_time:
            current_duration = time() - self.focus_start_time
            if current_duration < self.min_focus_duration:
                # Не переключаемся - слишком рано
                return False
        
        # Правило 5: Новый кандидат важнее?
        if candidate_focus and candidate_focus != self.current_focus:
            # Упрощенная проверка: просто переключаемся если кандидат другой
            # (можно добавить проверку важности с switching_cost)
            return True
        
        # По умолчанию - не переключаемся
        return False
    
    def _switch_focus(self, new_focus: Optional[str], reason: str) -> Optional[str]:
        """Переключить фокус на новый"""
        # Сохранить старый фокус в историю
        if self.current_focus and self.focus_start_time:
            duration = time() - self.focus_start_time
            self.focus_history.append({
                "focus": self.current_focus,
                "duration": duration,
                "reason": "switched"
            })
            
            # Ограничить историю
            if len(self.focus_history) > 20:
                self.focus_history.pop(0)
        
        # Установить новый фокус
        self.current_focus = new_focus
        self.focus_start_time = time()
        
        print(f"[ATTENTION] Новый фокус: '{new_focus[:50] if new_focus else 'None'}...' (причина: {reason})")
        
        return new_focus
    
    def _maintain_focus(self) -> Optional[str]:
        """Удержать текущий фокус"""
        return self.current_focus
    
    def _maintain_or_clear_focus(self) -> Optional[str]:
        """Удержать или очистить фокус если нет стимулов"""
        if self.current_focus and self.focus_start_time:
            duration = time() - self.focus_start_time
            
            # Если фокус слишком старый - очистить
            if duration > self.max_focus_duration * 2:
                print(f"[ATTENTION] Очистка старого фокуса ({duration:.1f}s)")
                self.current_focus = None
                self.focus_start_time = None
                return None
        
        return self.current_focus
    
    def get_focus_duration(self) -> float:
        """Получить длительность текущего фокуса"""
        if self.focus_start_time:
            return time() - self.focus_start_time
        return 0.0
    
    def get_focus_stats(self) -> Dict:
        """Статистика внимания"""
        current_duration = self.get_focus_duration()
        
        # Средняя длительность фокуса
        if self.focus_history:
            avg_duration = sum(f["duration"] for f in self.focus_history) / len(self.focus_history)
        else:
            avg_duration = 0.0
        
        # Количество переключений
        switch_count = len(self.focus_history)
        
        return {
            "current_focus": self.current_focus,
            "current_duration": round(current_duration, 2),
            "average_duration": round(avg_duration, 2),
            "switch_count": switch_count,
            "focus_history_size": len(self.focus_history)
        }
    
    def force_switch(self, reason: str = "manual"):
        """Принудительно переключить фокус на следующем цикле"""
        if self.current_focus:
            print(f"[ATTENTION] Запланировано переключение фокуса (причина: {reason})")
            # В следующем select_focus() с force_switch=True переключится


# Для обратной совместимости - старый простой метод
class SimpleAttentionSystem:
    """Старая версия для совместимости"""
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def select_focus(self, stimuli: List[Dict]) -> Optional[str]:
        if not stimuli:
            return None
        best = max(stimuli, key=lambda s: s.get("salience", 0))
        return best["content"] if best.get("salience", 0) >= self.threshold else None

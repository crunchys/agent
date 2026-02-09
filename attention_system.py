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
        
        # Отслеживание текущего фокуса
        self.current_focus = None
        self.focus_start_time = None
        self.focus_duration = 0.0
        
        # История фокусов
        self.focus_history = []
        
        # Параметры attention span
        self.min_focus_duration = 2.0  # Минимум 2 секунды
        self.max_focus_duration = 30.0  # Максимум 30 секунд
        
        # Switching cost
        self.switching_cost = 0.2
    
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
            force_switch: Принудительно переключить
        
        Returns:
            Содержимое фокуса или None
        """
        if not stimuli:
            return self._maintain_or_clear_focus()
        
        # 1. Bottom-up attention (от стимулов)
        bottom_up_focus = self._select_by_salience(stimuli)
        
        # 2. Top-down attention (от цели)
        top_down_focus = self._select_by_goal(stimuli, current_goal)
        
        # 3. Объединение
        candidate_focus = self._combine_attention(
            bottom_up_focus, 
            top_down_focus,
            stimuli
        )
        
        # 4. Проверка switching
        should_switch = self._should_switch(
            candidate_focus,
            force_switch
        )
        
        if should_switch:
            return self._switch_focus(candidate_focus, "new_stimulus")
        else:
            return self._maintain_focus()
    
    def _select_by_salience(self, stimuli: List[Dict]) -> Optional[Dict]:
        """Bottom-up: выбор по важности"""
        if not stimuli:
            return None
        
        best = max(stimuli, key=lambda s: s.get("salience", 0))
        
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
        
        goal_lower = current_goal.lower()
        
        for stimulus in stimuli:
            content = stimulus.get("content", "").lower()
            
            # Узнать о собеседнике
            if "узнать" in goal_lower or "вопрос" in goal_lower:
                if "content" in stimulus:
                    return stimulus
            
            # Поддержать разговор
            if "поддержать" in goal_lower or "разговор" in goal_lower:
                if stimulus.get("salience", 0) > 0.3:
                    return stimulus
            
            # Разобраться
            if "разобраться" in goal_lower:
                if stimulus.get("prediction_error", 0) > 0.5:
                    return stimulus
        
        return None
    
    def _combine_attention(
        self,
        bottom_up: Optional[Dict],
        top_down: Optional[Dict],
        stimuli: List[Dict]
    ) -> Optional[str]:
        """Объединение bottom-up и top-down"""
        if top_down and bottom_up:
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
        """Решить, нужно ли переключить фокус"""
        if force_switch:
            return True
        
        if self.current_focus is None:
            return True
        
        # Слишком долго
        if self.focus_start_time:
            current_duration = time() - self.focus_start_time
            if current_duration > self.max_focus_duration:
                print(f"[ATTENTION] Переключение: фокус слишком долго ({current_duration:.1f}s)")
                return True
        
        # Слишком короткий (стабильность)
        if self.focus_start_time:
            current_duration = time() - self.focus_start_time
            if current_duration < self.min_focus_duration:
                return False
        
        # Новый кандидат важнее
        if candidate_focus and candidate_focus != self.current_focus:
            return True
        
        return False
    
    def _switch_focus(self, new_focus: Optional[str], reason: str) -> Optional[str]:
        """Переключить фокус"""
        if self.current_focus and self.focus_start_time:
            duration = time() - self.focus_start_time
            self.focus_history.append({
                "focus": self.current_focus,
                "duration": duration,
                "reason": "switched"
            })
            
            if len(self.focus_history) > 20:
                self.focus_history.pop(0)
        
        self.current_focus = new_focus
        self.focus_start_time = time()
        
        if new_focus:
            print(f"[ATTENTION] Новый фокус: '{new_focus[:50]}...' (причина: {reason})")
        
        return new_focus
    
    def _maintain_focus(self) -> Optional[str]:
        """Удержать текущий фокус"""
        return self.current_focus
    
    def _maintain_or_clear_focus(self) -> Optional[str]:
        """Удержать или очистить фокус если нет стимулов"""
        if self.current_focus and self.focus_start_time:
            duration = time() - self.focus_start_time
            
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
        
        if self.focus_history:
            avg_duration = sum(f["duration"] for f in self.focus_history) / len(self.focus_history)
        else:
            avg_duration = 0.0
        
        switch_count = len(self.focus_history)
        
        return {
            "current_focus": self.current_focus,
            "current_duration": round(current_duration, 2),
            "average_duration": round(avg_duration, 2),
            "switch_count": switch_count,
            "focus_history_size": len(self.focus_history)
        }

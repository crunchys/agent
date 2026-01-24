from typing import List, Dict, Optional
from dataclasses import dataclass
from time import time
import random

@dataclass
class MemoryItem:
    content: str
    emotion: str
    valence: float
    arousal: float
    timestamp: float
    importance: float  # 0.0-1.0
    access_count: int = 0  # сколько раз вспоминали
    last_access: float = 0.0
    
    def decay_importance(self, decay_rate: float = 0.01):
        """Важность угасает со временем"""
        time_passed = time() - self.timestamp
        self.importance *= (1 - decay_rate * (time_passed / 3600))  # за час
        self.importance = max(0.0, self.importance)


class WorkingMemory:
    """
    Рабочая память - активна СЕЙЧАС (5-7 элементов)
    """
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[MemoryItem] = []
    
    def add(self, item: MemoryItem) -> Optional[MemoryItem]:
        """Добавить элемент, вернуть вытесненный если есть"""
        self.items.append(item)
        
        if len(self.items) > self.capacity:
            # Вытесняем НАИМЕНЕЕ важный (не первый!)
            least_important = min(self.items[:-1], key=lambda x: x.importance)
            self.items.remove(least_important)
            return least_important  # Вернуть для переноса в Short-term
        
        return None
    
    def get_context(self) -> List[MemoryItem]:
        """Получить контекст для мыслей"""
        return self.items[-3:]  # Последние 3 самые активные
    
    def clear(self):
        """Очистить (конец сессии)"""
        old_items = self.items.copy()
        self.items = []
        return old_items


class ShortTermMemory:
    """
    Кратковременная память - последние события (50-100 элементов)
    """
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.items: List[MemoryItem] = []
    
    def add(self, item: MemoryItem) -> Optional[MemoryItem]:
        """Добавить, вернуть вытесненный если нужна консолидация"""
        self.items.append(item)
        
        if len(self.items) > self.capacity:
            # Сортируем по важности
            self.items.sort(key=lambda x: x.importance, reverse=True)
            
            # Забываем наименее важный
            forgotten = self.items.pop()
            
            # Если забытый был ВАЖНЫМ (>0.7) - вернуть для Long-term
            if forgotten.importance > 0.7:
                return forgotten
        
        return None
    
    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        """Простой поиск по содержимому (можно улучшить через embeddings)"""
        # Простая эвристика: ищем по ключевым словам
        results = [item for item in self.items if any(word in item.content.lower() for word in query.lower().split())]
        
        # Сортируем по важности и свежести
        results.sort(key=lambda x: x.importance * (1 - (time() - x.timestamp) / 3600), reverse=True)
        
        return results[:k]
    
    def decay_all(self):
        """Угасание всех воспоминаний (вызывать периодически)"""
        for item in self.items:
            item.decay_importance()


class LongTermMemory:
    """
    Долговременная память - важные воспоминания, паттерны, знания
    """
    def __init__(self):
        self.episodic: List[MemoryItem] = []  # Важные события
        self.semantic: Dict[str, any] = {}  # Факты, знания (e.g., "собаки - животные")
        self.patterns: Dict[str, int] = {}  # Паттерны (e.g., "вопрос о хобби" -> 5 раз)
    
    def consolidate(self, item: MemoryItem):
        """
        Консолидация: сохранить важное событие или паттерн
        """
        # 1. Если очень важное - сохранить как эпизод
        if item.importance > 0.7 or abs(item.valence) > 0.6:
            self.episodic.append(item)
            print(f"[LTM] Консолидация эпизода: {item.content[:50]}... (важность: {item.importance:.2f})")
        
        # 2. Обнаружить паттерн (повторяющиеся темы)
        keywords = self._extract_keywords(item.content)
        for kw in keywords:
            self.patterns[kw] = self.patterns.get(kw, 0) + 1
            
            # Если паттерн повторился 5+ раз - это "знание"
            if self.patterns[kw] >= 5 and kw not in self.semantic:
                self.semantic[kw] = {"type": "pattern", "count": 5, "learned": time()}
                print(f"[LTM] Новое знание через паттерн: '{kw}'")
    
    def recall_similar(self, query: str, k: int = 3) -> List[MemoryItem]:
        """Вспомнить похожие эпизоды (можно улучшить через vector search)"""
        results = [item for item in self.episodic if any(word in item.content.lower() for word in query.lower().split())]
        
        # Сортируем по важности
        results.sort(key=lambda x: x.importance, reverse=True)
        
        # Обновляем access_count
        for item in results[:k]:
            item.access_count += 1
            item.last_access = time()
        
        return results[:k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Простое извлечение ключевых слов"""
        # Убрать стоп-слова
        stop_words = {"и", "в", "на", "с", "по", "из", "к", "о", "для", "что", "как", "это", "был"}
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        return keywords[:3]  # Топ-3


class IntegratedMemorySystem:
    """
    Интегрированная система памяти - управляет всеми уровнями
    """
    def __init__(self):
        self.working = WorkingMemory(capacity=7)
        self.short_term = ShortTermMemory(capacity=100)
        self.long_term = LongTermMemory()
    
    def process_event(self, event: Dict):
        """
        Обработать новое событие - распределить по уровням памяти
        """
        # 1. Вычислить важность
        valence = event.get("valence", 0.0)
        arousal = event.get("arousal", 0.0)
        prediction_error = event.get("prediction_error", 0.0)
        
        # Формула важности: эмоциональная интенсивность + новизна
        importance = (abs(valence) * 0.4 + arousal * 0.3 + prediction_error * 0.3)
        importance = min(1.0, importance)
        
        # 2. Создать MemoryItem
        memory_item = MemoryItem(
            content=event.get("focus", ""),
            emotion=event.get("emotion", "neutral"),
            valence=valence,
            arousal=arousal,
            timestamp=time(),
            importance=importance
        )
        
        # 3. Добавить в Working Memory
        displaced = self.working.add(memory_item)
        
        # 4. Если вытеснили - перенести в Short-term
        if displaced:
            forgotten = self.short_term.add(displaced)
            
            # 5. Если из Short-term вытеснили важное - консолидировать в Long-term
            if forgotten:
                self.long_term.consolidate(forgotten)
        
        return memory_item
    
    def get_context_for_thought(self) -> str:
        """
        Получить контекст для генерации мысли
        Комбинирует Working + релевантные из Short-term + Long-term
        """
        # Working memory (самое свежее)
        working_context = self.working.get_context()
        
        context_parts = []
        
        # Последние 3 из Working
        for item in working_context:
            context_parts.append(f"- {item.content[:50]} ({item.emotion})")
        
        return "\n".join(context_parts) if context_parts else "Нет активного контекста"
    
    def search_memories(self, query: str) -> Dict[str, List[MemoryItem]]:
        """
        Поиск по всем уровням памяти
        """
        return {
            "working": [item for item in self.working.items if query.lower() in item.content.lower()],
            "short_term": self.short_term.search(query, k=3),
            "long_term": self.long_term.recall_similar(query, k=2)
        }
    
    def end_session(self):
        """
        Конец сессии - перенести Working в Short-term, запустить консолидацию
        """
        print("[ПАМЯТЬ] Завершение сессии...")
        
        # Переносим всё из Working в Short-term
        for item in self.working.items:
            self.short_term.add(item)
        
        self.working.clear()
        
        # Угасание Short-term
        self.short_term.decay_all()
        
        print(f"[ПАМЯТЬ] STM: {len(self.short_term.items)} событий, LTM: {len(self.long_term.episodic)} эпизодов")

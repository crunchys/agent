import json
import os
from typing import List, Dict

class PersistentMemory:
    """
    Простая файловая память для агента:
    - Загружает события из JSON при инициализации
    - Сохраняет на диск после каждого изменения
    - Предоставляет методы store() и recent()
    """

    def __init__(self, path: str = "memory.json", autosave: bool = True):
        self.path = path
        self.autosave = autosave
        self.events: List[Dict] = []
        self._load_from_file()

    def _load_from_file(self):
        """Попытка загрузить память из файла, если он существует."""
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf‑8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.events = data
                    else:
                        self.events = []
            else:
                self.events = []
        except Exception as e:
            print(f"⚠️ Не удалось загрузить память из {self.path}: {e}")
            self.events = []

    def _save_to_file(self):
        """Сохранение текущей памяти в файл JSON."""
        try:
            with open(self.path, "w", encoding="utf‑8") as f:
                json.dump(self.events, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении памяти в {self.path}: {e}")

    def store(self, event: Dict):
        """Сохранить новое событие памяти."""
        self.events.append(event)
        if self.autosave:
            self._save_to_file()

    def recent(self, n: int = 5) -> List[Dict]:
        """
        Вернуть последние n событий памяти.
        Если событий меньше, вернуть все.
        """
        return self.events[-n:]

    def all(self) -> List[Dict]:
        """Вернуть всю память (для отладки или анализа)."""
        return self.events

    def clear(self):
        """Очистить память и сохранить пустой файл."""
        self.events = []
        if self.autosave:
            self._save_to_file()

    def save(self):
        """Вручную сохранить память в файл."""
        self._save_to_file()

    def load(self):
        """Вручную перезагрузить память из файла (заменяет текущую)."""
        self._load_from_file()

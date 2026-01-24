import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from time import time
import torch

class VectorMemory:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.events = []

    def add_event(self, event):
        # ИЗМЕНЕНО: Добавляем эмоцию в текст для поиска
        emotion = event.get('emotion', 'neutral')
        text = (
            f"Эмоция: {emotion}. "
            f"Фокус: {event.get('focus', '')}. "
            f"Мысль: {event.get('thought', '')}. "
            f"Arousal: {event.get('arousal', 0)}, Valence: {event.get('valence', 0)}."
        )
        emb = self.embedder.encode(text)
        self.index.add(emb.reshape(1, -1))
        self.events.append(event)

    def search(self, query, k=3):
        emb = self.embedder.encode(query)
        _, indices = self.index.search(emb.reshape(1, -1), k)
        valid_indices = [i for i in indices[0] if i != -1 and i < len(self.events)]
        return [self.events[i] for i in valid_indices]
    
    def search_by_emotion(self, emotion: str, k=5) -> List[Dict]:
        """
        НОВОЕ: Поиск событий по эмоции
        """
        matching = [e for e in self.events if e.get('emotion') == emotion]
        return matching[-k:] if matching else []

class EpisodicMemory:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.events: List[Dict] = []

    def store(self, event: Dict):
        self.events.append(event)
        if len(self.events) > self.capacity:
            self.events.pop(0)

    def recent(self, n: int = 5):
        return self.events[-n:]

class MetaReflection:
    def __init__(self, window: int = 5):
        self.window = window

    def reflect(self, memory: EpisodicMemory) -> Optional[Dict]:
        recent = [e for e in memory.events if e["type"] == "experience"][-self.window:]
        if len(recent) < self.window:
            return None
        
        avg_arousal = sum(e["arousal"] for e in recent) / self.window
        avg_valence = sum(e["valence"] for e in recent) / self.window
        
        # НОВОЕ: Анализ эмоциональных паттернов
        emotions = [e.get("emotion", "neutral") for e in recent]
        emotion_counts = {}
        for em in emotions:
            emotion_counts[em] = emotion_counts.get(em, 0) + 1
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        
        # Урок по правилам с учётом эмоций
        if avg_valence < -0.3:
            if dominant_emotion == "fear":
                lesson = "Из этого я понял, что такие ситуации вызывают страх - стоит быть осторожнее"
            elif dominant_emotion == "anger":
                lesson = "Из этого я понял, что такие темы вызывают раздражение - нужно избегать"
            else:
                lesson = "Из этого я понял, что подобные взаимодействия вредны, нужно избегать."
        elif avg_valence > 0.3:
            if dominant_emotion == "joy":
                lesson = "Из этого я понял, что такие темы приносят радость - стоит продолжать"
            else:
                lesson = "Из этого я понял, что такие темы полезны, стоит продолжать."
        else:
            lesson = "Из этого я понял, что ситуация нейтральна."
        
        return {
            "type": "meta_reflection",
            "time": round(time(), 2),
            "avg_arousal": round(avg_arousal, 3),
            "avg_valence": round(avg_valence, 3),
            "dominant_emotion": dominant_emotion,  # НОВОЕ
            "lesson": lesson
        }

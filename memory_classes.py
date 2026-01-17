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
        text = f"Фокус: {event.get('focus', '')}. Мысль: {event.get('thought', '')}. Arousal: {event.get('arousal', 0)}, Valence: {event.get('valence', 0)}."
        emb = self.embedder.encode(text)
        self.index.add(emb.reshape(1, -1))
        self.events.append(event)

    def search(self, query, k=3):
        emb = self.embedder.encode(query)
        _, indices = self.index.search(emb.reshape(1, -1), k)
        valid_indices = [i for i in indices[0] if i != -1 and i < len(self.events)]
        return [self.events[i] for i in valid_indices]

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
    def __init__(self, window: int = 5, model=None, tokenizer=None):
        self.window = window
        self.model = model
        self.tokenizer = tokenizer

    def reflect(self, memory: EpisodicMemory) -> Optional[Dict]:
        recent = [e for e in memory.events if e["type"] == "experience"][-self.window:]
        if len(recent) < self.window:
            return None
        
        avg_arousal = sum(e["arousal"] for e in recent) / self.window
        avg_valence = sum(e["valence"] for e in recent) / self.window
        
        events_summary = ", ".join(f"{e['focus']} (a:{e['arousal']:.2f}, v:{e['valence']:.2f}, thought:{e['thought'][:50]}...)" for e in recent)
        prompt = (
            "Анализируй эти события и извлеки ключевой урок или абстракцию: "
            f"{events_summary}. "
            "Урок должен быть кратким: 1-2 предложения на русском. "
            "Начни с 'Из этого я понял, что...'."
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones_like(inputs["input_ids"])
        output_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )
        lesson = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
        
        return {
            "type": "meta_reflection",
            "time": round(time(), 2),
            "avg_arousal": round(avg_arousal, 3),
            "avg_valence": round(avg_valence, 3),
            "lesson": lesson
        }

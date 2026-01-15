import json
from typing import List, Dict

class PersistentMemory:
    def __init__(self, filename="memory.json"):
        self.filename = filename
        self.events: List[Dict] = []
        self.world: Dict[str, Dict] = {}

    # ====== Episodic Memory ======
    def store_event(self, event: Dict):
        self.events.append(event)

    def recent_events(self, n=5):
        return self.events[-n:]

    # ====== World Model ======
    def update_object(self, name: str, properties: Dict = None, relations: Dict = None):
        if name not in self.world:
            self.world[name] = {"properties": {}, "relations": {}}
        if properties:
            self.world[name]["properties"].update(properties)
        if relations:
            self.world[name]["relations"].update(relations)

    def get_object(self, name: str):
        return self.world.get(name, {"properties": {}, "relations": {}})

    # ====== Save / Load ======
    def save(self):
        data = {"events": self.events, "world": self.world}
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.events = data.get("events", [])
            self.world = data.get("world", {})
        except FileNotFoundError:
            self.events = []
            self.world = {}

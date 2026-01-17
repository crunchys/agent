from typing import List, Dict, Optional
from collections import Counter

class EmotionSystem:
    def __init__(self, decay_rate: float = 0.05):
        self.decay_rate = decay_rate

    def apply_stimulus(self, state, stimulus: Dict):
        state.arousal = min(1.0, state.arousal + stimulus["intensity"])
        state.valence = max(-1.0, min(1.0, state.valence + stimulus["valence"]))
        return state

    def decay(self, state):
        state.arousal *= (1 - self.decay_rate)
        state.valence *= (1 - self.decay_rate)
        state.existence_threat *= (1 - self.decay_rate * 0.5)
        return state

    def apply_success_failure(self, state, is_success: bool):
        if is_success:
            state.valence += 0.2
            state.arousal += 0.1
        else:
            state.valence -= 0.3
            state.arousal += 0.2
            state.existence_threat += 0.1

    def update_existence_threat(self, state, threat_delta: float):
        state.existence_threat = min(1.0, max(0.0, state.existence_threat + threat_delta))
        if state.existence_threat > 0.5:
            state.arousal += 0.15
            state.valence -= 0.1

class PredictionErrorSystem:
    def __init__(self):
        self.history = Counter()

    def compute(self, stimulus: Dict) -> float:
        content = stimulus["content"]
        self.history[content] += 1
        total = sum(self.history.values())
        freq = self.history[content] / total if total > 0 else 0.0
        prediction_error = 1.0 - freq
        return round(prediction_error, 3)

class AttentionSystem:
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def select_focus(self, stimuli: List[Dict]) -> Optional[str]:
        if not stimuli:
            return None
        best = max(stimuli, key=lambda s: s["salience"])
        return best["content"] if best["salience"] >= self.threshold else None

class FutureExpectationSystem:
    def __init__(self):
        self.expectations: Dict[str, Dict[str, float]] = {}

    def predict(self, content: str):
        if content not in self.expectations:
            self.expectations[content] = {'prob': 0.5, 'valence': 0.0, 'arousal': 0.3}
        return self.expectations[content]

    def update(self, content: str, actual_valence: float, actual_arousal: float):
        exp = self.predict(content)
        lr = 0.1
        exp['valence'] += lr * (actual_valence - exp['valence'])
        exp['arousal'] += lr * (actual_arousal - exp['arousal'])
        exp['prob'] = min(1.0, max(0.0, exp['prob'] + lr * (1 - exp['prob'])))

    def curiosity(self, content: str):
        exp = self.predict(content)
        surprise = 1.0 - exp['prob']
        return surprise

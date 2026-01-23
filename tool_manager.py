from typing import Dict, List, Optional, Any
from memory_classes import VectorMemory
from agent_memory import PersistentMemory
import json

class ToolManager:
    """
    Система grounding - проверка мыслей на фактах из мира.
    Предотвращает галлюцинации через обращение к памяти и внешним источникам.
    """
    
    def __init__(self, vector_memory: VectorMemory, persistent_memory: PersistentMemory):
        self.vector_memory = vector_memory
        self.persistent_memory = persistent_memory
        self.tool_results_cache = {}
    
    def ground(self, stimuli: List[Dict]) -> List[Dict]:
        """
        Grounding: проверить стимулы на фактах из памяти.
        Возвращает дополнительные события для интеграции в state.
        """
        grounded_results = []
        
        for stimulus in stimuli:
            content = stimulus.get("content", "")
            
            # 1. Поиск в векторной памяти
            memory_result = self.search_memory(content)
            
            # 2. Проверка на известные факты
            fact_check = self.check_known_facts(content)
            
            # 3. Определение gaps в знаниях
            knowledge_gap = self.detect_knowledge_gap(content, memory_result)
            
            grounded_event = {
                "type": "grounding",
                "stimulus": content,
                "memory_hits": len(memory_result),
                "fact_verified": fact_check["verified"],
                "knowledge_gap": knowledge_gap,
                "confidence": self._compute_confidence(memory_result, fact_check)
            }
            
            grounded_results.append(grounded_event)
        
        return grounded_results
    
    def search_memory(self, query: str, k: int = 3) -> List[Dict]:
        """Поиск релевантных событий в векторной памяти"""
        if not query.strip():
            return []
        
        try:
            results = self.vector_memory.search(query, k=k)
            return results
        except Exception as e:
            print(f"⚠️ Ошибка поиска в памяти: {e}")
            return []
    
    def check_known_facts(self, content: str) -> Dict[str, Any]:
        """
        Проверка на известные факты в персистентной памяти.
        Возвращает: {"verified": bool, "source": str, "confidence": float}
        """
        # Поиск фактов в последних событиях
        recent_events = self.persistent_memory.recent(20)
        
        # Простая эвристика: если похожий контент был в прошлом
        for event in recent_events:
            if "focus" in event and event["focus"]:
                # Простое совпадение слов (можно улучшить)
                if any(word in content.lower() for word in event["focus"].lower().split()):
                    return {
                        "verified": True,
                        "source": "episodic_memory",
                        "confidence": 0.7
                    }
        
        return {
            "verified": False,
            "source": None,
            "confidence": 0.0
        }
    
    def detect_knowledge_gap(self, content: str, memory_results: List[Dict]) -> float:
        """
        Определить, насколько агент НЕ знает о теме (0.0 = знает, 1.0 = не знает).
        Это повысит curiosity и prediction_error.
        """
        if not memory_results:
            # Нет релевантных воспоминаний = большой gap
            return 1.0
        
        # Если есть воспоминания, но они старые или с низкой valence
        avg_valence = sum(r.get("valence", 0) for r in memory_results) / len(memory_results)
        
        # Gap обратно пропорционален количеству и качеству памяти
        gap = 1.0 - (len(memory_results) / 10.0)  # max 10 результатов = 0 gap
        gap += (0.5 if avg_valence < 0 else -0.2)  # негативная память = больше gap
        
        return max(0.0, min(1.0, gap))
    
    def _compute_confidence(self, memory_results: List[Dict], fact_check: Dict) -> float:
        """
        Вычислить уверенность агента в понимании темы.
        Низкая confidence → высокий prediction_error → мотивация учиться.
        """
        memory_confidence = min(1.0, len(memory_results) / 5.0)
        fact_confidence = fact_check["confidence"]
        
        # Среднее взвешенное
        confidence = (memory_confidence * 0.6 + fact_confidence * 0.4)
        return round(confidence, 3)
    
    def store_grounded_fact(self, fact: str, source: str, confidence: float):
        """
        Сохранить проверенный факт в память как grounded_event.
        Используется после успешного tool call или web search.
        """
        grounded_event = {
            "type": "grounded_fact",
            "fact": fact,
            "source": source,
            "confidence": confidence,
            "time": __import__("time").time()
        }
        
        self.persistent_memory.store(grounded_event)
        print(f"[GROUNDING] Факт сохранён: {fact[:50]}... (confidence: {confidence})")


# Placeholder для будущих расширений
class WebSearchTool:
    """Поиск в интернете (TODO: интеграция с API)"""
    def search(self, query: str) -> Dict:
        return {"error": "Not implemented yet"}


class CodeExecutionTool:
    """Выполнение кода для вычислений (TODO)"""
    def execute(self, code: str) -> Dict:
        return {"error": "Not implemented yet"}

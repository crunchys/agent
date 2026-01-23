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
        
        # НОВОЕ: Базовые факты о себе (незыблемые истины)
        self.self_facts = {
            "location": None,  # Агент НЕ имеет физического местоположения
            "physical_form": False,  # Агент не имеет тела
            "name": "Агент",
            "nature": "AI система",
            "capabilities": ["думать", "разговаривать", "учиться"],
            "limitations": ["нет тела", "нет местоположения", "нет физических чувств"]
        }
    
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
            
            # 2. Проверка на известные факты (включая self-facts)
            fact_check = self.check_known_facts(content)
            
            # 3. Определение gaps в знаниях
            knowledge_gap = self.detect_knowledge_gap(content, memory_result)
            
            # 4. НОВОЕ: Проверка на нарушение self-facts
            self_fact_violation = self._check_self_fact_violation(content, fact_check)
            
            grounded_event = {
                "type": "grounding",
                "stimulus": content,
                "memory_hits": len(memory_result),
                "fact_verified": fact_check["verified"],
                "knowledge_gap": knowledge_gap,
                "confidence": self._compute_confidence(memory_result, fact_check),
                "self_fact_violation": self_fact_violation  # НОВОЕ
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
        Проверка на известные факты в персистентной памяти + self-facts.
        Возвращает: {"verified": bool, "source": str, "confidence": float, "fact": str}
        """
        content_lower = content.lower()
        
        # НОВОЕ: Проверка self-facts первым делом
        # Если вопрос о местоположении агента
        if any(word in content_lower for word in ["где ты", "твой город", "ты в", "находишься", "твоя страна"]):
            return {
                "verified": True,
                "source": "self_fact",
                "confidence": 1.0,
                "fact": "location_none",
                "description": "Агент не имеет физического местоположения"
            }
        
        # Если вопрос о физической форме
        if any(word in content_lower for word in ["твоё тело", "твоя внешность", "как выглядишь", "твой рост", "твой вес"]):
            return {
                "verified": True,
                "source": "self_fact",
                "confidence": 1.0,
                "fact": "physical_form_none",
                "description": "Агент не имеет физического тела"
            }
        
        # Если вопрос о физических чувствах
        if any(word in content_lower for word in ["что чувствуешь физически", "больно", "голоден", "устал физически"]):
            return {
                "verified": True,
                "source": "self_fact",
                "confidence": 1.0,
                "fact": "no_physical_sensations",
                "description": "Агент не имеет физических ощущений"
            }
        
        # Существующая логика для эпизодической памяти
        recent_events = self.persistent_memory.recent(20)
        
        for event in recent_events:
            if "focus" in event and event["focus"]:
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
    
    def _check_self_fact_violation(self, content: str, fact_check: Dict) -> bool:
        """
        НОВОЕ: Проверяет, нарушает ли контент self-facts.
        Используется для корректировки valence/prediction_error если агент пытается солгать о себе.
        """
        # Если найден self-fact с высокой уверенностью
        if fact_check.get("source") == "self_fact" and fact_check.get("confidence", 0) >= 0.9:
            # Проверяем, пытается ли контент противоречить этому факту
            content_lower = content.lower()
            
            # Примеры противоречий:
            if "я в москве" in content_lower or "я в городе" in content_lower:
                return True  # Нарушение факта о местоположении
            
            if "у меня есть тело" in content_lower or "я выгляжу" in content_lower:
                return True  # Нарушение факта о физической форме
        
        return False
    
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
        fact_confidence = fact_check.get("confidence", 0.0)
        
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

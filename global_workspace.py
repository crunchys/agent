"""
Global Workspace Theory Implementation
=======================================

Реализация теории глобального рабочего пространства (Baars, 1988).

Основные компоненты:
1. WorkspaceMessage - сообщения от систем
2. SystemInterface - интерфейс для систем
3. WorkspaceState - состояние сознания
4. GlobalWorkspace - центральный координатор
5. Адаптеры для существующих систем

Автор: Cognitive Agent Project
Версия: 1.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from time import time
from abc import ABC, abstractmethod


# ============================================================================
# 1. БАЗОВЫЕ КЛАССЫ
# ============================================================================

@dataclass
class WorkspaceMessage:
    """
    Сообщение для Global Workspace.
    Каждая система посылает такие сообщения для доступа к сознанию.
    """
    source: str              # Откуда (e.g., "emotion_system")
    content: Any             # Содержимое
    salience: float          # Важность (0.0-1.0)
    urgency: float           # Срочность (0.0-1.0)
    message_type: str        # Тип (e.g., "emotion", "memory", "goal")
    timestamp: float = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time()
    
    def priority(self) -> float:
        """
        Приоритет для конкуренции.
        Формула: salience * 0.6 + urgency * 0.4
        """
        return self.salience * 0.6 + self.urgency * 0.4


@dataclass
class WorkspaceState:
    """
    Глобальное состояние workspace - "содержимое сознания".
    """
    # Основное содержимое
    conscious_content: List[WorkspaceMessage]
    
    # Интегрированное состояние
    integrated_emotion: Dict[str, float]
    integrated_focus: Optional[str]
    integrated_goal: Optional[str]
    integrated_memory: List[Dict]
    
    # Метаданные
    consciousness_level: float    # Уровень сознания (0-1)
    integration_score: float      # Степень интеграции (0-1)
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time()


class SystemInterface(ABC):
    """
    Базовый интерфейс для всех систем.
    Каждая система должна уметь генерировать сообщения и получать broadcasts.
    """
    
    @abstractmethod
    def generate_messages(self, context: Dict) -> List[WorkspaceMessage]:
        """Сгенерировать сообщения для workspace"""
        pass
    
    @abstractmethod
    def receive_broadcast(self, workspace_state: WorkspaceState):
        """Получить broadcast из workspace и обновить состояние"""
        pass
    
    @abstractmethod
    def get_system_name(self) -> str:
        """Имя системы"""
        pass


# ============================================================================
# 2. GLOBAL WORKSPACE (Главный класс)
# ============================================================================

class GlobalWorkspace:
    """
    Центральный координатор - "театр сознания".
    
    Цикл работы:
    1. COLLECT - собрать сообщения от систем
    2. COMPETE - конкуренция за доступ к сознанию
    3. INTEGRATE - интеграция в единое состояние
    4. BROADCAST - вещание всем системам
    """
    
    def __init__(
        self, 
        threshold: float = 0.5,        # Порог для "сознания"
        capacity: int = 3,              # Сколько одновременно в сознании
        integration_weight: float = 0.7 # Вес интеграции
    ):
        self.threshold = threshold
        self.capacity = capacity
        self.integration_weight = integration_weight
        
        # Зарегистрированные системы
        self.systems: Dict[str, SystemInterface] = {}
        
        # Текущее состояние
        self.current_state: Optional[WorkspaceState] = None
        
        # История (последние 100 состояний)
        self.state_history: List[WorkspaceState] = []
        
        # Статистика
        self.stats = {
            "total_messages": 0,
            "conscious_messages": 0,
            "broadcasts": 0,
            "total_cycles": 0,
        }
    
    def register_system(self, system: SystemInterface):
        """Зарегистрировать систему"""
        name = system.get_system_name()
        self.systems[name] = system
        print(f"[GW] ✓ Система зарегистрирована: {name}")
    
    def process_cycle(self, context: Dict) -> WorkspaceState:
        """
        Один цикл обработки (главная функция).
        
        Args:
            context: Контекст для систем (стимулы, текст пользователя и т.д.)
        
        Returns:
            WorkspaceState: Интегрированное состояние сознания
        """
        self.stats["total_cycles"] += 1
        
        # 1. COLLECT: Собрать сообщения
        all_messages = self._collect_messages(context)
        self.stats["total_messages"] += len(all_messages)
        
        # 2. COMPETE: Конкуренция
        conscious_messages = self._competition(all_messages)
        self.stats["conscious_messages"] += len(conscious_messages)
        
        # 3. INTEGRATE: Интеграция
        workspace_state = self._integration(conscious_messages, context)
        
        # 4. BROADCAST: Вещание
        self._broadcast(workspace_state)
        self.stats["broadcasts"] += 1
        
        # Сохранить состояние
        self.current_state = workspace_state
        self.state_history.append(workspace_state)
        
        # Ограничить историю
        if len(self.state_history) > 100:
            self.state_history.pop(0)
        
        return workspace_state
    
    def _collect_messages(self, context: Dict) -> List[WorkspaceMessage]:
        """Собрать сообщения от всех систем"""
        all_messages = []
        
        for system_name, system in self.systems.items():
            try:
                messages = system.generate_messages(context)
                all_messages.extend(messages)
            except Exception as e:
                print(f"[GW] ⚠️ Ошибка сбора от {system_name}: {e}")
        
        return all_messages
    
    def _competition(self, messages: List[WorkspaceMessage]) -> List[WorkspaceMessage]:
        """
        Конкуренция за доступ к сознанию.
        Отбираются сообщения с priority > threshold, максимум capacity.
        """
        if not messages:
            return []
        
        # Отфильтровать по threshold
        filtered = [m for m in messages if m.priority() >= self.threshold]
        
        # Сортировать по priority
        filtered.sort(key=lambda m: m.priority(), reverse=True)
        
        # Взять топ-capacity
        conscious = filtered[:self.capacity]
        
        if conscious:
            print(f"[GW] 🧠 Сознательное: {len(conscious)}/{len(messages)} сообщений")
            for msg in conscious:
                print(f"  → {msg.source}: {msg.message_type} (priority: {msg.priority():.2f})")
        
        return conscious
    
    def _integration(
        self, 
        conscious_messages: List[WorkspaceMessage],
        context: Dict
    ) -> WorkspaceState:
        """
        Интеграция сообщений в единое состояние.
        Это "синтез" информации - главная функция сознания.
        """
        
        # Инициализация
        integrated_emotion = {"arousal": 0.0, "valence": 0.0, "dominance": 0.0}
        integrated_focus = None
        integrated_goal = None
        integrated_memory = []
        
        # Счетчики для усреднения
        emotion_count = 0
        
        # Проход по сообщениям
        for msg in conscious_messages:
            # Интеграция эмоций
            if msg.message_type == "emotion":
                emotion_data = msg.content
                weight = msg.salience
                integrated_emotion["arousal"] += emotion_data.get("arousal", 0) * weight
                integrated_emotion["valence"] += emotion_data.get("valence", 0) * weight
                integrated_emotion["dominance"] += emotion_data.get("dominance", 0) * weight
                emotion_count += weight
            
            # Интеграция фокуса внимания (выбираем самый приоритетный)
            elif msg.message_type == "attention":
                if integrated_focus is None or msg.priority() > 0.7:
                    integrated_focus = msg.content.get("focus")
            
            # Интеграция целей (выбираем самую приоритетную)
            elif msg.message_type == "goal":
                if integrated_goal is None or msg.priority() > 0.6:
                    integrated_goal = msg.content.get("description")
            
            # Интеграция памяти (собираем все)
            elif msg.message_type == "memory":
                integrated_memory.append(msg.content)
        
        # Нормализация эмоций
        if emotion_count > 0:
            for key in integrated_emotion:
                integrated_emotion[key] /= emotion_count
        
        # Вычислить метрики
        consciousness_level = len(conscious_messages) / self.capacity if self.capacity > 0 else 0
        integration_score = self._compute_integration_score(conscious_messages)
        
        # Создать состояние
        state = WorkspaceState(
            conscious_content=conscious_messages,
            integrated_emotion=integrated_emotion,
            integrated_focus=integrated_focus,
            integrated_goal=integrated_goal,
            integrated_memory=integrated_memory,
            consciousness_level=consciousness_level,
            integration_score=integration_score,
            timestamp=time()
        )
        
        # Логирование
        print(f"[GW] 🎭 Интегрированное состояние:")
        print(f"  → Эмоция: arousal={integrated_emotion['arousal']:.2f}, "
              f"valence={integrated_emotion['valence']:.2f}")
        print(f"  → Фокус: {integrated_focus or 'нет'}")
        print(f"  → Цель: {integrated_goal or 'нет'}")
        print(f"  → Память: {len(integrated_memory)} элементов")
        print(f"  → Уровень сознания: {consciousness_level:.2f}")
        print(f"  → Интеграция: {integration_score:.2f}")
        
        return state
    
    def _broadcast(self, workspace_state: WorkspaceState):
        """
        Вещание состояния всем системам.
        Каждая система обновляется на основе глобального состояния.
        """
        for system_name, system in self.systems.items():
            try:
                system.receive_broadcast(workspace_state)
            except Exception as e:
                print(f"[GW] ⚠️ Ошибка broadcast к {system_name}: {e}")
    
    def _compute_integration_score(self, messages: List[WorkspaceMessage]) -> float:
        """
        Вычислить степень интеграции.
        Высокая интеграция = сообщения от разных систем.
        """
        if not messages:
            return 0.0
        
        # Уникальные источники
        unique_sources = len(set(m.source for m in messages))
        
        # Уникальные типы
        unique_types = len(set(m.message_type for m in messages))
        
        # Интеграция = разнообразие источников и типов
        max_sources = len(self.systems) if self.systems else 1
        max_types = 5  # emotion, memory, goal, attention, prediction
        
        source_diversity = unique_sources / max_sources
        type_diversity = unique_types / max_types
        
        integration = (source_diversity + type_diversity) / 2
        
        return min(1.0, integration)
    
    def get_stats(self) -> Dict:
        """Статистика работы workspace"""
        return {
            **self.stats,
            "registered_systems": len(self.systems),
            "current_consciousness_level": self.current_state.consciousness_level if self.current_state else 0,
            "current_integration": self.current_state.integration_score if self.current_state else 0,
            "avg_messages_per_cycle": self.stats["total_messages"] / max(1, self.stats["total_cycles"]),
            "avg_conscious_per_cycle": self.stats["conscious_messages"] / max(1, self.stats["total_cycles"]),
        }
    
    def get_integration_history(self, n: int = 10) -> List[float]:
        """Получить историю integration score"""
        return [s.integration_score for s in self.state_history[-n:]]
    
    def get_consciousness_history(self, n: int = 10) -> List[float]:
        """Получить историю consciousness level"""
        return [s.consciousness_level for s in self.state_history[-n:]]


# ============================================================================
# 3. АДАПТЕРЫ ДЛЯ СУЩЕСТВУЮЩИХ СИСТЕМ
# ============================================================================

class EmotionSystemAdapter(SystemInterface):
    """Адаптер для EmotionSystem"""
    
    def __init__(self, emotion_system, mental_state):
        self.emotion_system = emotion_system
        self.mental_state = mental_state
    
    def generate_messages(self, context: Dict) -> List[WorkspaceMessage]:
        """Генерировать сообщение об эмоциональном состоянии"""
        
        # Вычислить salience (важность эмоции)
        salience = abs(self.mental_state.valence) * 0.5 + self.mental_state.arousal * 0.5
        salience = min(1.0, salience)
        
        # Urgency зависит от threat
        urgency = self.mental_state.existence_threat
        
        # Текущая эмоция
        current_emotion = self.emotion_system.get_current_emotion(self.mental_state)
        
        message = WorkspaceMessage(
            source="emotion_system",
            content={
                "arousal": self.mental_state.arousal,
                "valence": self.mental_state.valence,
                "dominance": self.mental_state.dominance,
                "threat": self.mental_state.existence_threat,
                "emotion": current_emotion
            },
            salience=salience,
            urgency=urgency,
            message_type="emotion",
            timestamp=time()
        )
        
        return [message]
    
    def receive_broadcast(self, workspace_state: WorkspaceState):
        """Обновить эмоции на основе глобального состояния"""
        
        # Слабое влияние интегрированной эмоции на текущую
        integrated_emotion = workspace_state.integrated_emotion
        influence_strength = 0.1
        
        self.mental_state.valence += integrated_emotion.get("valence", 0) * influence_strength
        self.mental_state.arousal += integrated_emotion.get("arousal", 0) * influence_strength
        
        # Клэмпинг
        self.mental_state.valence = max(-1.0, min(1.0, self.mental_state.valence))
        self.mental_state.arousal = max(0.0, min(1.0, self.mental_state.arousal))
    
    def get_system_name(self) -> str:
        return "emotion_system"


class MemorySystemAdapter(SystemInterface):
    """Адаптер для IntegratedMemorySystem"""
    
    def __init__(self, memory_system):
        self.memory_system = memory_system
    
    def generate_messages(self, context: Dict) -> List[WorkspaceMessage]:
        """Генерировать сообщения о важных воспоминаниях"""
        
        messages = []
        
        # Взять топ-3 воспоминания из working memory
        working_context = self.memory_system.working.get_context()
        
        for item in working_context:
            # Salience = важность воспоминания
            salience = item.importance
            
            # Urgency = свежесть (чем свежее, тем срочнее)
            age = time() - item.timestamp
            urgency = max(0, 1.0 - age / 3600)  # За час угасает до 0
            
            message = WorkspaceMessage(
                source="memory_system",
                content={
                    "content": item.content,
                    "emotion": item.emotion,
                    "valence": item.valence,
                    "arousal": item.arousal,
                },
                salience=salience,
                urgency=urgency,
                message_type="memory",
                timestamp=time()
            )
            
            messages.append(message)
        
        return messages
    
    def receive_broadcast(self, workspace_state: WorkspaceState):
        """Обновить память на основе глобального состояния"""
        
        # Если в workspace есть важный фокус - усилить связанные воспоминания
        if workspace_state.integrated_focus:
            for item in self.memory_system.working.items:
                if workspace_state.integrated_focus.lower() in item.content.lower():
                    item.importance = min(1.0, item.importance + 0.1)
    
    def get_system_name(self) -> str:
        return "memory_system"


class PlannerAdapter(SystemInterface):
    """Адаптер для Planner"""
    
    def __init__(self, planner):
        self.planner = planner
    
    def generate_messages(self, context: Dict) -> List[WorkspaceMessage]:
        """Генерировать сообщения о текущих целях"""
        
        messages = []
        
        # Активные цели
        active_goals = [g for g in self.planner.goals if g.status == "active"]
        
        for goal in active_goals[:2]:  # Топ-2 цели
            # Salience = приоритет цели
            salience = goal.priority
            
            # Urgency = обратно пропорционален прогрессу
            urgency = 1.0 - goal.progress
            
            message = WorkspaceMessage(
                source="planning_system",
                content={
                    "description": goal.description,
                    "priority": goal.priority,
                    "progress": goal.progress,
                    "steps": goal.steps
                },
                salience=salience,
                urgency=urgency,
                message_type="goal",
                timestamp=time()
            )
            
            messages.append(message)
        
        return messages
    
    def receive_broadcast(self, workspace_state: WorkspaceState):
        """Обновить планы на основе глобального состояния"""
        
        # Если в workspace сильные негативные эмоции - снизить приоритет сложных целей
        valence = workspace_state.integrated_emotion.get("valence", 0)
        
        if valence < -0.3:
            for goal in self.planner.goals:
                if goal.priority > 0.7:  # Сложные цели
                    goal.priority *= 0.95  # Небольшое снижение
    
    def get_system_name(self) -> str:
        return "planning_system"


class AttentionSystemAdapter(SystemInterface):
    """Адаптер для AttentionSystem"""
    
    def __init__(self, attention_system, state):
        self.attention_system = attention_system
        self.state = state
    
    def generate_messages(self, context: Dict) -> List[WorkspaceMessage]:
        """Генерировать сообщение о текущем фокусе"""
        
        if not self.state.focus:
            return []
        
        # Salience = насколько важен фокус
        salience = 0.7  # Базовое значение
        
        # Urgency = всегда средняя для внимания
        urgency = 0.5
        
        message = WorkspaceMessage(
            source="attention_system",
            content={
                "focus": self.state.focus,
            },
            salience=salience,
            urgency=urgency,
            message_type="attention",
            timestamp=time()
        )
        
        return [message]
    
    def receive_broadcast(self, workspace_state: WorkspaceState):
        """Обновить внимание на основе глобального состояния"""
        
        # Если есть интегрированный фокус - использовать его
        if workspace_state.integrated_focus and workspace_state.integration_score > 0.6:
            self.state.focus = workspace_state.integrated_focus
    
    def get_system_name(self) -> str:
        return "attention_system"


class PredictionSystemAdapter(SystemInterface):
    """Адаптер для PredictionErrorSystem и FutureExpectationSystem"""
    
    def __init__(self, prediction_system, future_system, current_prediction_error: float = 0.0):
        self.prediction_system = prediction_system
        self.future_system = future_system
        self.current_prediction_error = current_prediction_error
    
    def generate_messages(self, context: Dict) -> List[WorkspaceMessage]:
        """Генерировать сообщение о prediction error"""
        
        # Salience = величина prediction error
        salience = min(1.0, self.current_prediction_error)
        
        # Urgency = также зависит от ошибки
        urgency = min(1.0, self.current_prediction_error * 0.8)
        
        if salience < 0.3:  # Не шлем сообщение если ошибка маленькая
            return []
        
        message = WorkspaceMessage(
            source="prediction_system",
            content={
                "prediction_error": self.current_prediction_error,
                "surprise": self.current_prediction_error > 0.7,
            },
            salience=salience,
            urgency=urgency,
            message_type="prediction",
            timestamp=time()
        )
        
        return [message]
    
    def receive_broadcast(self, workspace_state: WorkspaceState):
        """Обновить предсказания на основе глобального состояния"""
        # Пока ничего не делаем
        pass
    
    def get_system_name(self) -> str:
        return "prediction_system"


# ============================================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def create_workspace_summary(workspace_state: WorkspaceState) -> str:
    """
    Создать текстовую сводку состояния workspace для использования в промптах.
    """
    if not workspace_state:
        return "Нет интегрированного состояния"
    
    emotion = workspace_state.integrated_emotion
    
    summary = f"""Интегрированное состояние сознания:
- Эмоция: arousal={emotion['arousal']:.2f}, valence={emotion['valence']:.2f}, dominance={emotion['dominance']:.2f}
- Фокус внимания: {workspace_state.integrated_focus or 'нет'}
- Текущая цель: {workspace_state.integrated_goal or 'нет'}
- Активных воспоминаний: {len(workspace_state.integrated_memory)}
- Уровень сознания: {workspace_state.consciousness_level:.2f} (0-1)
- Интеграция систем: {workspace_state.integration_score:.2f} (0-1)
"""
    
    return summary


def print_workspace_stats(gw: GlobalWorkspace):
    """Красиво напечатать статистику workspace"""
    stats = gw.get_stats()
    
    print("\n" + "="*60)
    print("GLOBAL WORKSPACE СТАТИСТИКА")
    print("="*60)
    print(f"Зарегистрировано систем: {stats['registered_systems']}")
    print(f"Всего циклов: {stats['total_cycles']}")
    print(f"Всего сообщений: {stats['total_messages']}")
    print(f"Сознательных сообщений: {stats['conscious_messages']}")
    print(f"Broadcasts: {stats['broadcasts']}")
    print(f"\nСредних сообщений за цикл: {stats['avg_messages_per_cycle']:.1f}")
    print(f"Средних сознательных за цикл: {stats['avg_conscious_per_cycle']:.1f}")
    print(f"\nТекущий уровень сознания: {stats['current_consciousness_level']:.2f}")
    print(f"Текущая интеграция: {stats['current_integration']:.2f}")
    print("="*60 + "\n")


# ============================================================================
# 5. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    """
    Пример использования (для тестирования).
    Реальная интеграция происходит в agent_core.py
    """
    
    print("Global Workspace System v1.0")
    print("Пример использования:\n")
    
    # Создать workspace
    gw = GlobalWorkspace(threshold=0.5, capacity=3)
    
    # Создать mock системы для примера
    class MockEmotionSystem:
        def get_current_emotion(self, state):
            return "calm"
    
    class MockState:
        def __init__(self):
            self.arousal = 0.3
            self.valence = 0.1
            self.dominance = 0.0
            self.existence_threat = 0.0
            self.focus = "тестирование"
    
    # Создать адаптер
    mock_emotion = MockEmotionSystem()
    mock_state = MockState()
    emotion_adapter = EmotionSystemAdapter(mock_emotion, mock_state)
    
    # Зарегистрировать
    gw.register_system(emotion_adapter)
    
    # Запустить цикл
    context = {"test": True}
    workspace_state = gw.process_cycle(context)
    
    # Вывести результат
    print("\n" + create_workspace_summary(workspace_state))
    print_workspace_stats(gw)
    
    print("✓ Тест успешен!")

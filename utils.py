from llama_cpp import Llama
import os

def load_model_and_tokenizer(
    model_path="qwen2.5-14b-instruct-q4_k_m.gguf",
    n_gpu_layers=35,  # Сколько слоев на GPU (-1 = все)
    n_ctx=4096,       # Размер контекста
    n_threads=8,      # CPU threads
    verbose=True
):
    """
    Загрузка GGUF модели через llama.cpp
    
    Преимущества vs transformers:
    - Меньше памяти (~2-3x)
    - Быстрее генерация
    - Проще использовать
    """
    try:
        print("=" * 70)
        print("ЗАГРУЗКА МОДЕЛИ: llama.cpp")
        print("=" * 70)
        print(f"Модель: {model_path}")
        print(f"GPU layers: {n_gpu_layers}")
        print(f"Context size: {n_ctx}")
        print()
        
        if not os.path.exists(model_path):
            print(f"❌ Файл модели не найден: {model_path}")
            print("\nСкачай модель:")
            print("https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print("Загрузка модели...")
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # -1 = все на GPU
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
            n_batch=512,
        )
        
        print("✓ Модель загружена")
        print("=" * 70)
        
        # llama-cpp не использует отдельный tokenizer
        # возвращаем llm дважды для совместимости
        return llm, llm
        
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        raise

# Для совместимости с агентом
class LlamaCppTokenizer:
    """Обёртка для совместимости с transformers API"""
    def __init__(self, llm):
        self.llm = llm
        self.eos_token = "</s>"
        self.pad_token = "</s>"
        self.eos_token_id = 2
        self.pad_token_id = 2
    
    def __call__(self, text, return_tensors=None):
        # llama-cpp не нужны tensors
        return {"input_text": text}
    
    def decode(self, tokens, skip_special_tokens=True):
        # Для llama-cpp text уже декодирован
        return tokens if isinstance(tokens, str) else ""

if __name__ == "__main__":
    # Тест
    model_path = "qwen2.5-14b-instruct-q4_k_m.gguf"
    llm, _ = load_model_and_tokenizer(model_path)
    
    print("\nТест генерации...")
    prompt = "Привет! Как дела?"
    
    response = llm(
        prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        echo=False
    )
    
    print(f"\nПромпт: {prompt}")
    print(f"Ответ: {response['choices'][0]['text']}")
    print("\n✅ Тест успешен!")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct", hf_token=None):
    try:
        print("=" * 50)
        print("ПРОВЕРКА CUDA")
        print("=" * 50)
        print(f"CUDA доступна: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA версия: {torch.version.cuda}")
            print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Очистка кэша
            torch.cuda.empty_cache()
            
            device = "cuda"
            dtype = torch.float16  # НЕ bfloat16!
        else:
            print("⚠️ CUDA недоступна, используется CPU")
            device = "cpu"
            dtype = torch.float32
        
        print(f"Устройство: {device}")
        print(f"Тип данных: {dtype}")
        print("=" * 50)

        print("Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Токенизатор загружен")
        print(f"Загрузка модели {model_name}...")
        
        if device == "cuda":
            # Для GPU - загружаем БЕЗ device_map="auto"
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            # Явно переносим на GPU
            print("Перенос модели на GPU...")
            model = model.to(device)
        else:
            # Для CPU - обычная загрузка
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True
            )

        model.eval()
        
        # Проверка
        actual_device = next(model.parameters()).device
        print(f"✓ Модель загружена на: {actual_device}")
        
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"✓ Использовано памяти GPU: {mem_used:.2f} GB")
        
        print("=" * 50)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        raise

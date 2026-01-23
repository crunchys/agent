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
            
            # Очистка кэша GPU перед загрузкой
            torch.cuda.empty_cache()
            
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            
            print(f"Память GPU (всего): {mem_total:.2f} GB")
            print(f"Память GPU (зарезервировано): {mem_reserved:.2f} GB")
            print(f"Память GPU (использовано): {mem_allocated:.2f} GB")
            print(f"Память GPU (свободно): {mem_total - mem_allocated:.2f} GB")
            
            device = "cuda"
            # Используем float16 вместо bfloat16 для совместимости
            dtype = torch.float16
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
            padding_side="left",
            token=hf_token
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✓ Токенизатор загружен")
        print(f"Загрузка модели {model_name}...")
        print("(это может занять 1-2 минуты при первой загрузке)")
        
        # Используем dtype вместо torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,  # Исправлено: было torch_dtype
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=hf_token
        )

        model.eval()
        
        # Проверка устройства
        actual_device = next(model.parameters()).device
        print(f"✓ Модель загружена на: {actual_device}")
        
        if torch.cuda.is_available():
            mem_allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            print(f"✓ Использовано памяти GPU: {mem_allocated_after:.2f} GB")
        
        print("=" * 50)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        raise

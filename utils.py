import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import psutil
import gc

def load_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-14B-Instruct",
    hf_token=None,
    use_4bit=True,
    use_cpu_offload=False,  # Auto-detect если True
    gpu_memory_limit="7GB",  # Для offloading
    cpu_memory_limit="16GB"  # Для offloading
):
    """
    Загрузка Qwen2.5-14B-Instruct с автоматическим выбором стратегии:
    - Если влезает в GPU → полностью на GPU (быстро)
    - Если не влезает → CPU offloading (медленно, но работает)
    
    Args:
        model_name: Имя модели (по умолчанию Qwen2.5-14B-Instruct)
        hf_token: HuggingFace токен
        use_4bit: 4-bit квантизация (рекомендуется True)
        use_cpu_offload: Принудительный CPU offload (False = auto-detect)
        gpu_memory_limit: Лимит GPU для offloading
        cpu_memory_limit: Лимит CPU RAM для offloading
    
    Returns:
        model, tokenizer
    """
    try:
        print("=" * 70)
        print("ЗАГРУЗКА МОДЕЛИ: Qwen2.5-14B-Instruct")
        print("=" * 70)
        
        # Проверка системы
        print("\n[1/6] Проверка системы...")
        if not torch.cuda.is_available():
            print("❌ CUDA недоступна! Модель не запустится.")
            raise RuntimeError("CUDA required")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cpu_memory = psutil.virtual_memory().total / 1024**3
        
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
        print(f"✓ CPU Memory: {cpu_memory:.1f} GB")
        
        if gpu_memory < 6:
            print("⚠️  Мало GPU памяти! Рекомендуется минимум 8GB")
        
        if cpu_memory < 12 and use_cpu_offload:
            print("⚠️  Мало CPU памяти для offloading! Рекомендуется 16GB+")
        
        # Очистка памяти
        print("\n[2/6] Очистка памяти...")
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ Память очищена")
        
        # Загрузка токенизатора
        print("\n[3/6] Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Токенизатор загружен")
        
        # Выбор стратегии загрузки
        print(f"\n[4/6] Определение стратегии загрузки...")
        
        # Оценка необходимой памяти для 14B в 4-bit
        estimated_vram = 7.5 if use_4bit else 14.0
        
        if not use_cpu_offload and gpu_memory >= estimated_vram:
            strategy = "gpu_only"
            print(f"✓ Стратегия: GPU ONLY (нужно ~{estimated_vram:.1f}GB, доступно {gpu_memory:.1f}GB)")
        else:
            strategy = "cpu_offload"
            print(f"✓ Стратегия: CPU OFFLOADING (GPU: {gpu_memory:.1f}GB недостаточно для {estimated_vram:.1f}GB)")
            print(f"  → Часть модели будет на CPU")
            print(f"  → Генерация будет медленнее (~2-3x)")
        
        # Загрузка модели
        print(f"\n[5/6] Загрузка модели...")
        print(f"Модель: {model_name}")
        print("⚠️  Первая загрузка займет 5-15 минут (скачивание)")
        print()
        
        if strategy == "gpu_only":
            model = _load_gpu_only(model_name, hf_token, use_4bit)
        else:
            model = _load_with_offload(
                model_name, 
                hf_token, 
                use_4bit,
                gpu_memory_limit,
                cpu_memory_limit
            )
        
        print("✓ Модель загружена")
        
        model.eval()
        
        # Финальная проверка памяти
        print("\n[6/6] Проверка использования памяти...")
        _print_memory_usage()
        
        print("\n" + "=" * 70)
        print("✅ ЗАГРУЗКА ЗАВЕРШЕНА")
        print("=" * 70)
        print()
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ЗАГРУЗКИ: {e}")
        import traceback
        traceback.print_exc()
        raise


def _load_gpu_only(model_name, hf_token, use_4bit):
    """Загрузка полностью в GPU (быстрая генерация)"""
    print("Загрузка в GPU...")
    
    if use_4bit:
        print("  • 4-bit квантизация (NF4)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
    else:
        print("  • FP16 (без квантизации)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
    
    return model


def _load_with_offload(model_name, hf_token, use_4bit, gpu_limit, cpu_limit):
    """Загрузка с CPU offloading (медленная генерация, но работает)"""
    print("Загрузка с CPU OFFLOADING...")
    print(f"  • GPU limit: {gpu_limit}")
    print(f"  • CPU limit: {cpu_limit}")
    
    # Создать папку для offloading
    offload_folder = "offload_cache"
    os.makedirs(offload_folder, exist_ok=True)
    
    # Конфигурация памяти
    max_memory = {
        0: gpu_limit,      # Первая GPU
        "cpu": cpu_limit   # CPU RAM
    }
    
    if use_4bit:
        print("  • 4-bit квантизация + offloading")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=True,
        )
    else:
        print("  • FP16 + offloading")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=True,
        )
    
    return model


def _print_memory_usage():
    """Вывод информации об использовании памяти"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory:")
        print(f"  • Allocated: {allocated:.2f} GB")
        print(f"  • Reserved:  {reserved:.2f} GB")
        print(f"  • Total:     {total:.2f} GB")
        print(f"  • Free:      {total - reserved:.2f} GB")
        
        if allocated > total * 0.9:
            print("  ⚠️  Память почти заполнена! Риск OOM.")
        elif allocated > total * 0.8:
            print("  ⚠️  Память заполнена на 80%+")
        else:
            print("  ✓ Память в норме")
    
    # CPU Memory
    cpu_mem = psutil.virtual_memory()
    cpu_used = cpu_mem.used / 1024**3
    cpu_total = cpu_mem.total / 1024**3
    cpu_percent = cpu_mem.percent
    
    print(f"\nCPU Memory:")
    print(f"  • Used:  {cpu_used:.2f} GB / {cpu_total:.2f} GB ({cpu_percent:.1f}%)")
    
    if cpu_percent > 90:
        print("  ⚠️  CPU память почти заполнена!")


# ============================================================================
# АЛЬТЕРНАТИВНАЯ ВЕРСИЯ: Разные модели
# ============================================================================

def load_alternative_model(model_choice="14b", hf_token=None):
    """
    Загрузка альтернативных моделей с автоматическими настройками.
    
    Варианты:
    - "14b" → Qwen2.5-14B-Instruct (рекомендуется, ~7GB)
    - "7b"  → Qwen2.5-7B-Instruct (быстрая, ~4.5GB)
    - "32b" → Qwen2.5-32B-Instruct (мощная, нужен offload)
    """
    
    models = {
        "7b": {
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "use_4bit": True,
            "use_cpu_offload": False,
            "gpu_limit": "5GB",
            "cpu_limit": "12GB"
        },
        "14b": {
            "name": "Qwen/Qwen2.5-14B-Instruct",
            "use_4bit": True,
            "use_cpu_offload": False,  # Auto-detect
            "gpu_limit": "7GB",
            "cpu_limit": "16GB"
        },
        "32b": {
            "name": "Qwen/Qwen2.5-32B-Instruct",
            "use_4bit": True,
            "use_cpu_offload": True,  # Принудительно
            "gpu_limit": "6GB",
            "cpu_limit": "20GB"
        }
    }
    
    if model_choice not in models:
        raise ValueError(f"Неизвестная модель: {model_choice}. Доступны: {list(models.keys())}")
    
    config = models[model_choice]
    
    print(f"Загрузка предустановки: {model_choice.upper()}")
    print(f"Модель: {config['name']}")
    print()
    
    return load_model_and_tokenizer(
        model_name=config["name"],
        hf_token=hf_token,
        use_4bit=config["use_4bit"],
        use_cpu_offload=config["use_cpu_offload"],
        gpu_memory_limit=config["gpu_limit"],
        cpu_memory_limit=config["cpu_limit"]
    )


# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

if __name__ == "__main__":
    """Тест загрузки модели"""
    import time
    
    print("\n" + "="*70)
    print("ТЕСТ ЗАГРУЗКИ МОДЕЛИ")
    print("="*70 + "\n")
    
    # Тест загрузки
    start = time.time()
    model, tokenizer = load_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        use_4bit=True,
        use_cpu_offload=False  # Auto-detect
    )
    end = time.time()
    
    print(f"\n✓ Загрузка заняла: {end - start:.1f} секунд ({(end-start)/60:.1f} минут)")
    
    # Тест генерации
    print("\n" + "="*70)
    print("ТЕСТ ГЕНЕРАЦИИ")
    print("="*70 + "\n")
    
    prompt = "Привет! Расскажи кратко что ты умеешь."
    print(f"Промпт: {prompt}\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Генерация...")
    start_gen = time.time()
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_gen = time.time()
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    
    print(f"Ответ: {response}\n")
    print(f"✓ Генерация заняла: {end_gen - start_gen:.1f} секунд")
    
    # Финальная проверка памяти
    print("\n" + "="*70)
    print("ФИНАЛЬНАЯ ПРОВЕРКА ПАМЯТИ")
    print("="*70 + "\n")
    _print_memory_usage()
    
    print("\n" + "="*70)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("="*70)

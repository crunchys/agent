import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

def load_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # 7B вместо 14B!
    hf_token=None,
    use_4bit=True
):
    """
    Загрузка Qwen2.5-7B-Instruct - влезает полностью в 8GB GPU!
    """
    try:
        print("=" * 70)
        print("ЗАГРУЗКА МОДЕЛИ")
        print("=" * 70)
        print(f"Модель: {model_name}")
        
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA недоступна")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Очистка памяти
        torch.cuda.empty_cache()
        gc.collect()
        print("\n✓ Память очищена")
        
        # Токенизатор
        print("\nЗагрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Токенизатор загружен")
        
        # Модель
        print(f"\nЗагрузка модели в GPU...")
        print("⚠️ Первая загрузка займет 3-7 минут (скачивание)")
        print()
        
        if use_4bit:
            print("Используется 4-bit квантизация (NF4)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",  # Автоматически на GPU
                trust_remote_code=True,
                token=hf_token,
                low_cpu_mem_usage=True,
            )
        else:
            print("Используется FP16 (без квантизации)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
                low_cpu_mem_usage=True,
            )
        
        print("✓ Модель загружена в GPU")
        model.eval()
        
        # Статистика памяти
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"\nGPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Free:      {gpu_memory - reserved:.2f} GB")
        
        if allocated > gpu_memory * 0.9:
            print("  ⚠️ Память почти заполнена!")
        else:
            print("  ✓ Память в норме")
        
        print("=" * 70)
        print("✅ ЗАГРУЗКА ЗАВЕРШЕНА")
        print("=" * 70)
        print()
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ЗАГРУЗКИ: {e}")
        import traceback
        traceback.print_exc()
        raise


# Для отладки
if __name__ == "__main__":
    import time
    
    print("Тест загрузки Qwen2.5-7B-Instruct")
    print()
    
    start = time.time()
    model, tokenizer = load_model_and_tokenizer()
    end = time.time()
    
    print(f"\n✓ Загрузка заняла: {end - start:.1f} секунд")
    
    # Тест генерации
    print("\n" + "="*70)
    print("ТЕСТ ГЕНЕРАЦИИ")
    print("="*70)
    
    prompt = "Привет! Как дела?"
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
    
    print(f"Ответ: {response}")
    print(f"\n✓ Генерация заняла: {end_gen - start_gen:.1f} секунд")
    print("\n✅ ВСЁ РАБОТАЕТ!")

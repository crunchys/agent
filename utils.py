import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",  # GPTQ версия
    hf_token=None
):
    """Загрузка GPTQ квантизированной модели"""
    print("=" * 50)
    print("ЗАГРУЗКА GPTQ МОДЕЛИ")
    print("=" * 50)
    print(f"Модель: {model_name}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Загрузка модели (GPTQ 4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        max_memory={
            0: "7GiB",
            "cpu": "12GiB"
        }
    )
    
    model.eval()
    
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✓ VRAM использовано: {mem_used:.2f} GB")
    
    print("✓ Модель загружена!")
    print("=" * 50)
    
    return model, tokenizer

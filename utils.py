import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-14B-Instruct-AWQ",
    hf_token=None
):
    """Загрузка AWQ квантизированной модели"""
    print("=" * 50)
    print("ЗАГРУЗКА AWQ МОДЕЛИ")
    print("=" * 50)
    print(f"Модель: {model_name}")
    
    # Очистка памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Токенизатор
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Модель с разделением GPU/CPU
    print("Загрузка модели (AWQ 4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
        max_memory={
            0: "7GiB",      # GPU - оставляем 1GB запас
            "cpu": "12GiB"  # CPU offload
        }
    )
    
    model.eval()
    
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✓ VRAM использовано: {mem_used:.2f} GB")
    
    print("=" * 50)
    print("✓ Модель загружена!")
    print("=" * 50)
    
    return model, tokenizer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct", hf_token=None):
    try:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Квантизация для снижения памяти (применяем всегда, чтобы экономить VRAM на GPU или RAM на CPU)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",  # Автоматически использует GPU, если доступен
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config  # Применяем квантизацию всегда
        )

        model.eval()
        print(f"Модель загружена на {device}")
        return model, tokenizer
    except Exception as e:
        print("Ошибка загрузки модели:", e)
        raise

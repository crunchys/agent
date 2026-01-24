import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model_and_tokenizer(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    hf_token=None,
    use_4bit=True  # НОВОЕ: 4-bit квантизация
):
    try:
        print("=" * 50)
        print("ЗАГРУЗКА МОДЕЛИ")
        print("=" * 50)
        print(f"Модель: {model_name}")
        print(f"CUDA доступна: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            torch.cuda.empty_cache()
            device = "cuda"
        else:
            print("⚠️ CUDA недоступна, используется CPU")
            device = "cpu"
            use_4bit = False  # CPU не поддерживает 4-bit
        
        print("=" * 50)

        # Загрузка токенизатора
        print("Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Токенизатор загружен")
        print(f"Загрузка модели {model_name}...")
        
        if device == "cuda" and use_4bit:
            # 4-bit квантизация для экономии памяти
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            print("Загрузка с 4-bit квантизацией...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
            print("✓ Модель загружена с 4-bit квантизацией")
            
        elif device == "cuda":
            # FP16 без квантизации
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
            print("✓ Модель загружена в FP16")
        else:
            # CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token
            )
            print("✓ Модель загружена на CPU")

        model.eval()
        
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

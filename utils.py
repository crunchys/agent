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
            device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            print("⚠️ CUDA недоступна, используется CPU")
            device = "cpu"
            dtype = torch.float32
        
        print(f"Устройство: {device}")
        print(f"Тип данных: {dtype}")
        print("=" * 50)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Загрузка модели {model_name}...")
        
        # Попытка с квантизацией (только для GPU)
        if device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config
                )
                print("✓ Модель загружена с 4-bit квантизацией")
                
            except Exception as e:
                print(f"⚠️ Ошибка квантизации: {e}")
                print("Загрузка без квантизации...")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("✓ Модель загружена без квантизации")
        else:
            # CPU без квантизации
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("✓ Модель загружена на CPU")

        model.eval()
        
        # Проверка, где модель находится
        if hasattr(model, 'device'):
            print(f"Модель на устройстве: {model.device}")
        
        print("=" * 50)
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        raise

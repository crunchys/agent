import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

def load_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-14B-Instruct",
    hf_token=None,
    use_4bit=True
):
    try:
        print("=" * 70)
        print("ЗАГРУЗКА МОДЕЛИ")
        print("=" * 70)
        print(f"Модель: {model_name}")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна")
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        
        print("\nЗагрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Токенизатор загружен")
        
        print("\nЗагрузка модели с CPU offloading...")
        print("⚠️ Это займет 5-10 минут при первом запуске")
        
        offload_folder = "offload_cache"
        os.makedirs(offload_folder, exist_ok=True)
        
        max_memory = {
            0: "6GB",      # GPU
            "cpu": "12GB"  # CPU
        }
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
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
        
        print("✓ Модель загружена")
        model.eval()
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\nGPU Memory: {allocated:.2f} GB")
        print("=" * 70)
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        raise

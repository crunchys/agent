import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
from huggingface_hub import snapshot_download
import os

class ExLlamaWrapper:
    """Обёртка для совместимости с существующим кодом"""
    
    def __init__(self, model, tokenizer, generator, cache):
        self._model = model
        self._tokenizer = tokenizer
        self._generator = generator
        self._cache = cache
        self.device = torch.device("cuda")
    
    def generate(self, input_ids, attention_mask=None, max_new_tokens=100, 
                 do_sample=True, temperature=0.7, top_p=0.9, **kwargs):
        """Совместимость с transformers API"""
        # Декодируем input_ids обратно в текст
        prompt = self._tokenizer.decode(input_ids[0].tolist())
        
        # Настройки семплера
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_p = top_p
        
        # Генерация
        self._generator.warmup()
        output = self._generator.generate_simple(
            prompt,
            settings,
            max_new_tokens,
            seed=None
        )
        
        # Кодируем обратно в токены для совместимости
        output_ids = self._tokenizer.encode(output)
        return torch.tensor([output_ids], device=self.device)
    
    def eval(self):
        pass


class ExLlamaTokenizerWrapper:
    """Обёртка токенизатора для совместимости"""
    
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token_id = tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
    
    def __call__(self, text, return_tensors="pt", **kwargs):
        ids = self._tokenizer.encode(text)
        input_ids = torch.tensor([ids], device="cuda")
        return {"input_ids": input_ids}
    
    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids)
    
    def encode(self, text):
        return self._tokenizer.encode(text)


def load_model_and_tokenizer(
    model_name="turboderp/Qwen2.5-14B-Instruct-exl2",  # ExLlamaV2 версия
    revision="4.0bpw",  # 4-bit, ~8GB
    hf_token=None
):
    """Загрузка ExLlamaV2 модели"""
    print("=" * 50)
    print("ЗАГРУЗКА EXLLAMAV2 МОДЕЛИ")
    print("=" * 50)
    
    # Скачиваем модель
    print(f"Скачивание {model_name} ({revision})...")
    model_path = snapshot_download(
        repo_id=model_name,
        revision=revision,
        token=hf_token
    )
    print(f"Модель в: {model_path}")
    
    # Конфиг
    config = ExLlamaV2Config(model_path)
    config.max_seq_len = 4096
    
    # Модель
    print("Загрузка модели...")
    model = ExLlamaV2(config)
    model.load()
    
    # Кэш
    cache = ExLlamaV2Cache(model, max_seq_len=4096, lazy=True)
    
    # Токенизатор
    tokenizer = ExLlamaV2Tokenizer(config)
    
    # Генератор
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✓ VRAM использовано: {mem_used:.2f} GB")
    
    print("✓ Модель загружена!")
    print("=" * 50)
    
    # Возвращаем обёртки для совместимости
    model_wrapper = ExLlamaWrapper(model, tokenizer, generator, cache)
    tokenizer_wrapper = ExLlamaTokenizerWrapper(tokenizer)
    
    return model_wrapper, tokenizer_wrapper

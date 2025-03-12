class TokenizerWrapper:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Для Qwen

    def encode(self, text, **kwargs):
        return self.tokenizer(text, return_tensors="pt", **kwargs)
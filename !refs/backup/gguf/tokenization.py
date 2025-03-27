from transformers import AutoTokenizer
import torch
import logging

logger = logging.getLogger(__name__)

class TokenizerWrapper:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text, **kwargs):
        # Возвращаем тензор input_ids
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
            **kwargs
        )
        return encoding['input_ids']

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            **kwargs
        )
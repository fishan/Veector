# device/src/model_manager.py
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from .tensors import create_tensor
from .core import Veector

class ModelManager:
    def __init__(self, veector, model_dir, cache_dir="data/local_cache"):
        self.veector = veector
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_space = {}
        self.cache = {}
        self.hidden_size = 2048
        self.vocab_size = 32256  # Размер словаря DeepSeek
        self.num_layers = 24  # Количество слоев (проверьте точное значение для DeepSeek 1.3B)

    def load_pre_split_model(self, model_name, model_path, max_layers=None):
        if max_layers is None:
            max_layers = self.num_layers
        for file in self.model_dir.glob("**/*.pt"):
            if "layers" in file.name:
                layer_num = int(file.name.split(".")[2])
                if layer_num >= max_layers:
                    continue
            tensor_id = file.stem
            role = (
                "attention_weights" if "self_attn" in file.name else
                "mlp_weights" if "mlp" in file.name else
                "embeddings" if "embed_tokens" in file.name else
                "lm_head" if "lm_head" in file.name else
                "norm" if "norm" in file.name else
                "unknown"
            )
            metadata = {
                "path": str(file.relative_to(self.model_dir)),
                "role": role,
                "shape": torch.load(file, weights_only=True).shape
            }
            self.veector.db.save_tensor_metadata(tensor_id, metadata)
            self.model_space[(model_name, 0, (0, 0, 0))] = metadata

    def _get_cache_size(self):
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt")) / (1024 * 1024)
        return total_size

    def _cleanup(self, max_size_mb=1024):
        while self._get_cache_size() > max_size_mb:
            oldest_file = min(self.cache_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
            tensor_id = oldest_file.stem
            os.remove(oldest_file)
            del self.cache[tensor_id]

    def get_tensor(self, tensor_id):
        cached_path = self.cache_dir / f"{tensor_id}.pt"
        metadata = self.veector.db.get_tensor_metadata(tensor_id)
        if not metadata:
            raise ValueError(f"Tensor {tensor_id} not found in metadata")
        
        source_path = self.model_dir / metadata["path"]
        if tensor_id not in self.cache:
            if not cached_path.exists():
                print(f"Loading tensor {tensor_id} from {source_path} to {cached_path}")
                torch.save(torch.load(source_path, weights_only=True), cached_path)
                self._cleanup()
            self.cache[tensor_id] = cached_path
        else:
            print(f"Using cached tensor {tensor_id} from {cached_path}")
        return torch.load(cached_path, weights_only=True)

    def perform_inference(self, model_name, input_ids, max_length=20):
        # Эмбеддинги
        embed_tensor = self.get_tensor("model.embed_tokens.weight")
        hidden_states = F.embedding(input_ids, embed_tensor)
        print(f"After embedding: {hidden_states.shape}")

        # Проход через все слои
        for layer in range(self.num_layers):
            # Self-Attention
            q_proj = self.get_tensor(f"model.layers.{layer}.self_attn.q_proj.weight")
            k_proj = self.get_tensor(f"model.layers.{layer}.self_attn.k_proj.weight")
            v_proj = self.get_tensor(f"model.layers.{layer}.self_attn.v_proj.weight")
            o_proj = self.get_tensor(f"model.layers.{layer}.self_attn.o_proj.weight")
            norm1 = self.get_tensor(f"model.layers.{layer}.input_layernorm.weight")

            # Нормализация перед вниманием
            hidden_states = F.layer_norm(hidden_states, (self.hidden_size,), weight=norm1)

            q = hidden_states @ q_proj.T
            k = hidden_states @ k_proj.T
            v = hidden_states @ v_proj.T

            attn_scores = q @ k.transpose(-2, -1) / (self.hidden_size ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = attn_weights @ v
            attn_output = attn_output @ o_proj.T

            # Остаточная связь
            hidden_states = hidden_states + attn_output
            print(f"After self_attention layer {layer}: {hidden_states.shape}")

            # Feed-Forward
            gate_proj = self.get_tensor(f"model.layers.{layer}.mlp.gate_proj.weight")
            up_proj = self.get_tensor(f"model.layers.{layer}.mlp.up_proj.weight")
            down_proj = self.get_tensor(f"model.layers.{layer}.mlp.down_proj.weight")
            norm2 = self.get_tensor(f"model.layers.{layer}.post_attention_layernorm.weight")

            # Нормализация перед FFN
            hidden_states = F.layer_norm(hidden_states, (self.hidden_size,), weight=norm2)

            ff_intermediate = F.gelu(hidden_states @ gate_proj.T) * (hidden_states @ up_proj.T)
            ff_output = ff_intermediate @ down_proj.T

            # Остаточная связь
            hidden_states = hidden_states + ff_output
            print(f"After feed_forward layer {layer}: {hidden_states.shape}")

        # Финальная нормализация
        final_norm = self.get_tensor("model.norm.weight")
        hidden_states = F.layer_norm(hidden_states, (self.hidden_size,), weight=final_norm)

        # Генерация текста
        lm_head = self.get_tensor("lm_head.weight")
        output_ids = []
        current_input = input_ids

        for _ in range(max_length):
            logits = hidden_states @ lm_head.T
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids.append(next_token.item())
            current_input = torch.cat([current_input, next_token], dim=1)
            hidden_states = F.embedding(current_input, embed_tensor)
            
            for layer in range(self.num_layers):
                q = hidden_states @ q_proj.T
                k = hidden_states @ k_proj.T
                v = hidden_states @ v_proj.T
                attn_scores = q @ k.transpose(-2, -1) / (self.hidden_size ** 0.5)
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_output = attn_weights @ v
                attn_output = attn_output @ o_proj.T
                hidden_states = hidden_states + attn_output

                ff_intermediate = F.gelu(hidden_states @ gate_proj.T) * (hidden_states @ up_proj.T)
                ff_output = ff_intermediate @ down_proj.T
                hidden_states = hidden_states + ff_output

        # Очистка кэша
        for tensor_id in self.cache.copy():
            os.remove(self.cache[tensor_id])
            del self.cache[tensor_id]

        return torch.tensor(output_ids, dtype=torch.long)
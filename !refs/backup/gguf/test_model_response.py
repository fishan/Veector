import torch
from transformers import AutoTokenizer
from virtual_space import VirtualSpace, ModelDispatcher
from model_manager import ModelManager
from core import Veector
import logging
import gc
import psutil

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Возвращает объём используемой памяти в МБ."""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024**2
    gpu_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    return ram_mb, gpu_mb

class Observer:
    def __init__(self, virtual_space, tokenizer):
        self.virtual_space = virtual_space
        self.tokenizer = tokenizer
        self.device = virtual_space.dispatcher.device
        self.dispatcher = virtual_space.dispatcher
        self.virtual_matrix = virtual_space.virtual_matrix

    def process_layer(self, hidden_states, layer_idx, input_ids):
        """Обрабатывает один слой и возвращает новые hidden_states и токены внимания."""
        logger.info(f"Processing layer {layer_idx}")
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory before layer {layer_idx}: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")
        logger.info(f"Hidden states: shape {hidden_states.shape}, tokens {hidden_states.size(1)}, memory {hidden_states.element_size() * hidden_states.nelement() / 1024**2:.2f} MB")

        # Self-attention по частям
        q_proj = self.dispatcher.assemble_layer_tensor(layer_idx, "self_attn_q_proj_weight", (self.dispatcher.hidden_size, self.dispatcher.hidden_size))
        q = torch.matmul(hidden_states, q_proj.t())
        del q_proj
        torch.cuda.empty_cache()
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory after q_proj: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        k_proj = self.dispatcher.assemble_layer_tensor(layer_idx, "self_attn_k_proj_weight", (self.dispatcher.key_dim, self.dispatcher.hidden_size))
        k = torch.matmul(hidden_states, k_proj.t())
        del k_proj
        torch.cuda.empty_cache()
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory after k_proj: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        v_proj = self.dispatcher.assemble_layer_tensor(layer_idx, "self_attn_v_proj_weight", (self.dispatcher.key_dim, self.dispatcher.hidden_size))
        v = torch.matmul(hidden_states, v_proj.t())
        del v_proj
        torch.cuda.empty_cache()
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory after v_proj: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        head_dim = self.dispatcher.hidden_size // self.dispatcher.num_attention_heads
        head_dim_kv = self.dispatcher.key_dim // self.dispatcher.num_key_value_heads
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        q = q.view(batch_size, seq_len, self.dispatcher.num_attention_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.dispatcher.num_key_value_heads, head_dim_kv).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.dispatcher.num_key_value_heads, head_dim_kv).transpose(1, 2)

        k = k.repeat_interleave(self.dispatcher.num_attention_heads // self.dispatcher.num_key_value_heads, dim=1)
        v = v.repeat_interleave(self.dispatcher.num_attention_heads // self.dispatcher.num_key_value_heads, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dispatcher.hidden_size)

        # Извлекаем токены внимания для последнего токена
        last_token_weights = attn_weights[:, :, -1, :]  # [batch, heads, seq_len]
        top_k = 5  # Показываем топ-5 токенов с наибольшим вниманием
        top_weights, top_indices = torch.topk(last_token_weights, k=top_k, dim=-1)
        attended_tokens = [input_ids[0, idx].item() for idx in top_indices[0, 0]]  # Токены для первой головы

        del q, k, v, attn_scores
        torch.cuda.empty_cache()
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory after attention: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        o_proj = self.dispatcher.assemble_layer_tensor(layer_idx, "self_attn_o_proj_weight", (self.dispatcher.hidden_size, self.dispatcher.hidden_size))
        hidden_states = torch.matmul(attn_output, o_proj.t())
        del o_proj, attn_output
        torch.cuda.empty_cache()
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory after o_proj: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        # MLP по частям
        gate_proj = self.dispatcher.assemble_layer_tensor(layer_idx, "mlp_gate_proj_weight", (self.dispatcher.intermediate_size, self.dispatcher.hidden_size))
        gate = torch.matmul(hidden_states, gate_proj.t())
        del gate_proj
        torch.cuda.empty_cache()
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory after gate_proj: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        up_proj = self.dispatcher.assemble_layer_tensor(layer_idx, "mlp_up_proj_weight", (self.dispatcher.intermediate_size, self.dispatcher.hidden_size))
        up = torch.matmul(hidden_states, up_proj.t())
        del up_proj
        torch.cuda.empty_cache()
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory after up_proj: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        mlp_output = gate * up
        del gate, up
        torch.cuda.empty_cache()
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory after mlp_output: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        down_proj = self.dispatcher.assemble_layer_tensor(layer_idx, "mlp_down_proj_weight", (self.dispatcher.hidden_size, self.dispatcher.intermediate_size))
        hidden_states = torch.matmul(mlp_output, down_proj.t())
        del down_proj, mlp_output
        torch.cuda.empty_cache()
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory after down_proj: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        logger.info(f"Layer {layer_idx} processed, hidden_states shape: {hidden_states.shape}")
        return hidden_states, attended_tokens

    def decode_hidden_states(self, hidden_states):
        """Декодирует hidden_states через выходной слой (только для финального результата)."""
        ram_mb, gpu_mb = get_memory_usage()
        logger.info(f"Memory before decoding: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        output_blocks = sorted(self.dispatcher.get_output_blocks(), key=lambda x: int(x.split("_block")[1].split(".")[0]))
        logits = torch.zeros(hidden_states.size(0), 1, self.dispatcher.vocab_size, dtype=torch.float16, device=self.device)

        for block_key in output_blocks:
            block = self.dispatcher.load_block(block_key)
            block_logits = torch.matmul(hidden_states[:, -1:, :], block.t())
            start_idx = int(block_key.split("_block")[1].split(".")[0]) * block.shape[0]
            end_idx = start_idx + block.shape[0]
            logits[:, :, start_idx:end_idx] = block_logits
            del block, block_logits
            torch.cuda.empty_cache()

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        decoded_text = self.tokenizer.decode(next_token[0, 0].item(), skip_special_tokens=True)
        del logits, probs
        torch.cuda.empty_cache()
        return decoded_text

    def generate(self, input_text, max_length=50, temperature=0.7):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        hidden_states = self.virtual_matrix.embedding(input_ids, f"{self.virtual_space.current_model}_embed")
        logger.info(f"Initial embedded shape: {hidden_states.shape}, tokens: {hidden_states.size(1)}, memory: {hidden_states.element_size() * hidden_states.nelement() / 1024**2:.2f} MB")

        # Проходим через каждый слой по очереди
        for layer_idx in range(self.dispatcher.num_layers):
            hidden_states, attended_tokens = self.process_layer(hidden_states, layer_idx, input_ids)
            attended_text = [self.tokenizer.decode([token], skip_special_tokens=True) for token in attended_tokens]
            print(f"Layer {layer_idx} attended tokens: {attended_text} (ids: {attended_tokens})")

            # Очистка памяти перед следующим слоем
            gc.collect()
            torch.cuda.empty_cache()
            ram_mb, gpu_mb = get_memory_usage()
            logger.info(f"Memory after layer {layer_idx}: RAM {ram_mb:.2f} MB, GPU {gpu_mb:.2f} MB")

        # Финальный результат через выходной слой
        final_text = self.decode_hidden_states(hidden_states)
        print(f"Final output: {final_text}")
        return final_text

# Пример использования
if __name__ == "__main__":
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    metadata_path = f"/workspaces/Veector/data/blocks/{model_name}/{model_name}_metadata.json"
    vocab_size = 151936
    hidden_size = 1536
    num_layers = 28
    num_attention_heads = 12
    intermediate_size = 8960
    key_dim = 256
    num_key_value_heads = 2
    tokenizer = AutoTokenizer.from_pretrained(f"deepseek-ai/{model_name}")

    # Инициализация VirtualSpace
    veector = Veector(use_memory=False, ipfs_enabled=False)
    model_manager = ModelManager(veector, ipfs_enabled=False)
    virtual_space = VirtualSpace(veector, use_ipfs=False, model_manager=model_manager)
    virtual_space.tokenizer = tokenizer
    
    virtual_space.switch_model(
        model_name=model_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        key_dim=key_dim,
        num_key_value_heads=num_key_value_heads
    )

    # Создание Observer
    observer = Observer(virtual_space, tokenizer)
    input_text = "Привет, как дела?"
    response = observer.generate(input_text)
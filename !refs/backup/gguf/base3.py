# test_full_model.py
import torch
from transformers import AutoTokenizer
from virtual_space import VirtualSpace
from model_manager import ModelManager
from core import Veector
import logging
import gc
import psutil

# Nastroika logirovaniya
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Poluchenie tekushchego ispol'zovaniya pamyati."""
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024**2
    gpu_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    return ram_mb, gpu_mb

class FullModelTest:
    def __init__(self):
        # Inicializaciya komponentov
        self.veector = Veector(use_memory=False, ipfs_enabled=False)
        self.model_manager = ModelManager(self.veector, ipfs_enabled=False)
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.virtual_space = VirtualSpace(
            veector=self.veector,
            use_ipfs=False,
            model_manager=self.model_manager
        )
        
        # Parametry modeli
        self.model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
        self.vocab_size = 151936
        self.hidden_size = 1536
        self.num_layers = 28
        self.num_attention_heads = 12
        self.intermediate_size = 8960
        self.key_dim = 256
        self.num_key_value_heads = 2
        
        # Perenapravlenie switch_model v virtual_space
        self.virtual_space.switch_model(
            model_name=self.model_name,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            key_dim=self.key_dim,
            num_key_value_heads=self.num_key_value_heads
        )
        logger.info(f"Переключено на модель: {self.model_name}")

    def process_input(self, text):
        """Obrabotka vhodnogo teksta cherez model'."""
        # Tokenizaciya vvoda
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if torch.any(input_ids >= self.vocab_size):
            raise ValueError(f"Входные данные содержат значения, превышающие vocab_size ({self.vocab_size})")
        
        batch_size, seq_len = input_ids.shape
        logger.info(f"Input shape: {input_ids.shape}, tokens: {seq_len}")
        
        # Log tokennye dannye
        tokens = [self.tokenizer.decode([token.item()]) for token in input_ids[0]]
        logger.info(f"Input tokens: {tokens}")
        
        # Embedding
        with torch.no_grad():
            hidden_states = self.virtual_space.virtual_matrix.embedding(
                input_ids, f"{self.model_name}_embed"
            )
            logger.info(f"Embeddings shape: {hidden_states.shape}")
            
            # Obrabotka cherez vse sloi
            for layer_idx in range(self.num_layers):
                logger.info(f"Processing layer {layer_idx}")
                ram_before, _ = get_memory_usage()
                logger.info(f"Memory before layer {layer_idx}: RAM {ram_before:.2f} MB")
                
                # Samovnimanie
                q = self.virtual_space.virtual_matrix.linear(
                    hidden_states, f"{self.model_name}_layer{layer_idx}_self_attn_q_proj_weight", 
                    self.hidden_size, self.hidden_size
                )
                k = self.virtual_space.virtual_matrix.linear(
                    hidden_states, f"{self.model_name}_layer{layer_idx}_self_attn_k_proj_weight", 
                    self.hidden_size, self.hidden_size
                )
                v = self.virtual_space.virtual_matrix.linear(
                    hidden_states, f"{self.model_name}_layer{layer_idx}_self_attn_v_proj_weight", 
                    self.hidden_size, self.hidden_size
                )
                
                attn_weights = torch.softmax(
                    torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5), dim=-1
                )
                attn_output = torch.matmul(attn_weights, v)
                hidden_states = hidden_states + attn_output
                
                # Log samovnimaniya
                logger.info(f"Layer {layer_idx} attention weights max: {attn_weights.max().item():.4f}")
                logger.info(f"Layer {layer_idx} attention output shape: {attn_output.shape}")
                
                # MLP
                gate = self.virtual_space.virtual_matrix.linear(
                    hidden_states, f"{self.model_name}_layer{layer_idx}_mlp_gate_proj_weight", 
                    self.intermediate_size, self.hidden_size
                )
                up = self.virtual_space.virtual_matrix.linear(
                    hidden_states, f"{self.model_name}_layer{layer_idx}_mlp_up_proj_weight", 
                    self.intermediate_size, self.hidden_size
                )
                mlp_output = gate * up
                down = self.virtual_space.virtual_matrix.linear(
                    mlp_output, f"{self.model_name}_layer{layer_idx}_mlp_down_proj_weight", 
                    self.hidden_size, self.intermediate_size
                )
                hidden_states = hidden_states + down
                
                # Log MLP
                logger.info(f"Layer {layer_idx} MLP gate output max: {gate.max().item():.4f}")
                logger.info(f"Layer {layer_idx} MLP up output max: {up.max().item():.4f}")
                logger.info(f"Layer {layer_idx} MLP down output shape: {down.shape}")
                
                # Optimizaciya pamyati
                del q, k, v, attn_weights, attn_output, gate, up, mlp_output, down
                gc.collect()
                torch.cuda.empty_cache()
                
                ram_after, _ = get_memory_usage()
                logger.info(f"Memory after layer {layer_idx}: RAM {ram_after:.2f} MB")
            
            # Final'noe dekodirovanie
            output_blocks = sorted(
                [k for k in self.virtual_space.dispatcher.metadata.keys() if k.startswith(f"{self.model_name}_output")],
                key=lambda x: int(x.split("_block")[1])
            )
            logits = torch.zeros(batch_size, seq_len, self.vocab_size, dtype=torch.float16)
            
            for block_key in output_blocks:
                block = self.virtual_space.dispatcher.load_block(block_key)
                block_out_size, block_in_size = block.shape
                block_idx = int(block_key.split("_block")[1])
                start_row = block_idx * block_out_size
                end_row = start_row + block_out_size
                logits[:, :, start_row:end_row] = torch.matmul(hidden_states, block.t())
                del block
                gc.collect()
                torch.cuda.empty_cache()
            
            predicted_ids = torch.argmax(logits, dim=-1)
            output_tokens = [self.tokenizer.decode([token_id.item()]) for token_id in predicted_ids[0]]
            output_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            # Log vyhodnyh dannyh
            logger.info(f"Predicted token IDs: {predicted_ids.tolist()}")
            logger.info(f"Predicted tokens: {output_tokens}")
            logger.info(f"Final output: {output_text}")
            return output_text

if __name__ == "__main__":
    tester = FullModelTest()
    input_text = "Как дела?"
    output = tester.process_input(input_text)
    print(f"Ответ модели: {output}")
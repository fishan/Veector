import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import gc
import psutil
import random
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class TokenAgent(nn.Module):
    def __init__(self, dispatcher, token_idx, hidden_size, rank):
        super().__init__()
        self.dispatcher = dispatcher
        self.device = dispatcher.device
        self.token_idx = token_idx
        self.hidden_size = hidden_size
        self.rank = rank
        self.attn_scorer = nn.Linear(hidden_size, 1, dtype=torch.float16)
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.float16)

    def process_attention(self, hidden_states, layer_idx, active_rooms):
        # ... (ostal'nyj kod ostalsya bez izmenenij)
        return o_out[:, self.token_idx:self.token_idx+1]

class Observer(nn.Module):
    def __init__(self, dispatcher, max_layers=28, top_k=10):
        super().__init__()
        # ... (inicializaciya ostalas' bez izmenenij)
        
    def forward(self, input_ids, tokenizer, return_top_k=False):
        with torch.no_grad():
            hidden_states = self.dispatcher.virtual_matrix.embedding(
                input_ids, f"{self.dispatcher.model_name}_embed").to(torch.float16)
            
            logger.info(f"Embeddings shape: {hidden_states.shape}")
            strategy = self.classify_query(hidden_states)
            active_layers = self.select_layers(hidden_states, strategy)
            logger.info(f"Active layers: {active_layers}")

            for layer_idx in active_layers:
                hidden_states = self.process_layer(hidden_states, layer_idx)
                hidden_states = self.norm(hidden_states)
                
                # Log intermediate states through tokenizer
                if layer_idx % 5 == 0:
                    decoded = tokenizer.decode(hidden_states.argmax(dim=-1).cpu().numpy().tolist())
                    logger.info(f"Intermediate state at layer {layer_idx}: {decoded[:100]}...")

                self.clear_memory()

            if return_top_k:
                next_token_hidden = hidden_states[:, -1, :]
                output_blocks = sorted(
                    [key for key in self.dispatcher.metadata.keys() if "_output" in key],
                    key=lambda x: int(x.split("_block")[1]))
                all_logits = []
                
                for block_key in output_blocks:
                    block = self.dispatcher.load_block(block_key).to(torch.float16)
                    logits = torch.matmul(next_token_hidden, block.t())
                    all_logits.append(logits)
                    del block
                    self.clear_memory()
                    
                next_token_logits = torch.cat(all_logits, dim=-1)
                top_k_logits, top_k_indices = torch.topk(next_token_logits, k=self.top_k, dim=-1)
                probs = F.softmax(top_k_logits / 0.7, dim=-1)
                
                valid_token = False
                while not valid_token:
                    next_token_id = top_k_indices[0, torch.multinomial(probs, 1).squeeze()]
                    decoded = tokenizer.decode([next_token_id.item()])
                    if self.validate_output(decoded):
                        valid_token = True
                
                logger.info(f"Next token ID: {next_token_id.item()}, decoded: {decoded}")
                return next_token_id
            return hidden_states

    def validate_output(self, text):
        """Proverka logichnosti vykhodnogo teksta"""
        if any(token in text for token in ["<unk>", "[UNK]"]):
            return False
        return True

    def clear_memory(self):
        self.dispatcher.virtual_matrix.clear_cache()
        gc.collect()
        logger.info(f"Memory cleanup, RAM: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

# V osnovnom kode generacii otvetov
def generate_response(observer, input_text, tokenizer, max_steps=5):
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(observer.device)
    logger.info(f"Input IDs shape: {input_ids.shape}")
    generated_ids = input_ids.clone()
    
    for step in range(max_steps):
        with torch.no_grad():
            next_token_id = observer.forward(generated_ids, tokenizer, return_top_k=True)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            current_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
            logger.info(f"Step {step}: Current text: {current_text}")
            
            # Podrobnyj log teksta na kazhdom shage
            tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
            logger.info(f"Tokens evolution: {tokens}")
            
            if next_token_id.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
    
    response = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    logger.info(f"Generated response: {response}")
    return response



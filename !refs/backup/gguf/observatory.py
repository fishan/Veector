import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import gc
import psutil
from typing import List, Tuple

logger = logging.getLogger(__name__)

class TokenTracker:
    def __init__(self, vocab_size: int, max_active_tokens: int = 5, max_streams: int = 10, decay_rate: float = 0.9):
        self.vocab_size = vocab_size
        self.max_active_tokens = max_active_tokens
        self.max_streams = max_streams
        self.device = torch.device("cpu")
        self.decay_rate = decay_rate  # Коэффициент затухания значимости токенов
        self.token_scores = {}  # Словарь для хранения значимости токенов: {token_id: score}
        self.layer_history = []  # История значимости токенов по слоям

    def update(self, token_scores: torch.Tensor, layer_idx: int, parent_stream_idx: int) -> List[int]:
        """
        Обновляет значимость токенов на основе текущих scores и истории.

        :param token_scores: Тензор с весами внимания или логитами для токенов (shape: [seq_len, vocab_size]).
        :param layer_idx: Индекс текущего слоя.
        :param parent_stream_idx: Индекс "потока" (для поддержки нескольких контекстов).
        :return: Список активных токенов.
        """
        # Инициализация значимости токенов на первом слое
        if layer_idx == 0:
            self.token_scores = {}
            self.layer_history = []

        # Обновляем значимость токенов
        token_scores = token_scores.cpu().detach()  # Переносим на CPU для экономии памяти
        for token_id in range(self.vocab_size):
            if token_id in self.token_scores:
                # Затухание значимости для токенов из предыдущих слоёв
                self.token_scores[token_id] *= self.decay_rate
            score = token_scores[token_id].item() if token_id < token_scores.shape[0] else 0.0
            if score > 0:  # Игнорируем токены с нулевой значимостью
                self.token_scores[token_id] = self.token_scores.get(token_id, 0.0) + score

        # Сохраняем историю значимости для текущего слоя
        self.layer_history.append(dict(self.token_scores))

        # Выбираем топ-K активных токенов
        sorted_tokens = sorted(self.token_scores.items(), key=lambda x: x[1], reverse=True)
        active_tokens = [token_id for token_id, score in sorted_tokens[:self.max_active_tokens]]
        logger.info(f"Layer {layer_idx}: Active tokens: {active_tokens}, Top scores: {[self.token_scores[t] for t in active_tokens]}")
        return active_tokens

    def get_relevance_mask(self, seq_len: int) -> torch.Tensor:
        """
        Создаёт маску релевантности токенов для текущего контекста.

        :param seq_len: Длина последовательности.
        :return: Тензор маски (shape: [seq_len]), где 1 — релевантный токен, 0 — нерелевантный.
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        active_tokens = self.get_active_tokens()
        for i in range(seq_len):
            mask[i] = i in active_tokens  # Проверяем, является ли токен активным
        return mask

    def get_active_tokens(self) -> List[int]:
        """
        Возвращает список активных токенов.
        """
        sorted_tokens = sorted(self.token_scores.items(), key=lambda x: x[1], reverse=True)
        return [token_id for token_id, score in sorted_tokens[:self.max_active_tokens]]

class Observer(nn.Module):
    def __init__(self, dispatcher, tokenizer, max_layers: int = 28, top_k: int = 10, temp_threshold: float = 0.1):
        super().__init__()
        self.dispatcher = dispatcher
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.hidden_size = dispatcher.hidden_size
        self.vocab_size = dispatcher.vocab_size
        self.num_layers = min(dispatcher.num_layers, max_layers)
        self.num_attention_heads = dispatcher.num_attention_heads
        self.key_dim = dispatcher.key_dim
        self.num_key_value_heads = dispatcher.num_key_value_heads
        self.top_k = top_k
        self.temperature_decay = 0.95
        self.query_scorer = nn.Linear(self.hidden_size, 3, dtype=torch.float16).to(self.device)
        self.layer_scorer = nn.Linear(self.hidden_size, self.num_layers, dtype=torch.float16).to(self.device)
        self.room_scorer = nn.Linear(self.hidden_size, self.hidden_size // 256, dtype=torch.float16).to(self.device)
        self.attn_scorer = nn.Linear(self.hidden_size, 1, dtype=torch.float16).to(self.device)
        self.token_tracker = TokenTracker(self.vocab_size, max_active_tokens=10, decay_rate=0.9)
        self.block_cache = {}
        # Добавляем "шпионов" — сеть для оценки релевантности токенов
        self.relevance_scorer = nn.Linear(self.hidden_size + 1, 1, dtype=torch.float16).to(self.device)  # +1 для attention weights
        self.to(self.device)

    def _embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.device, dtype=torch.float16)
        
        block_size = 4096
        num_blocks = (self.vocab_size + block_size - 1) // block_size
        
        for i in range(num_blocks):
            block_key = f"{self.dispatcher.model_name}_embed_block{i}"
            block = self._load_block_with_cache(block_key)
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, self.vocab_size)
            
            mask = (input_ids >= start_idx) & (input_ids < end_idx)
            if mask.any():
                local_ids = (input_ids - start_idx).clamp(min=0, max=block.shape[0] - 1)
                local_embed = block[local_ids]
                hidden_states[mask] = local_embed[mask]
        
        if torch.isnan(hidden_states).any():
            logger.error(f"NaN in embedded hidden_states: {hidden_states[:, :5, :5]}")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)
        
        logger.info(f"Embedded input shape: {hidden_states.shape}, values: {hidden_states[:, :5, :5]}")
        return hidden_states

    def classify_query(self, hidden_states: torch.Tensor) -> str:
        pooled = hidden_states.mean(dim=[0, 1])
        scores = self.query_scorer(pooled).softmax(dim=-1)
        entropy = -(scores * torch.log(scores + 1e-10)).sum().item()
        strategy = "deep" if entropy > 2.0 else "medium" if entropy > 1.0 else "light"
        logger.info(f"Classified as {strategy} strategy (entropy: {entropy:.2f})")
        return strategy

    def select_layers(self, hidden_states: torch.Tensor, strategy: str) -> List[int]:
        layer_scores = self.layer_scorer(hidden_states.mean(dim=1)).sigmoid().squeeze(0)
        num_active = {"light": 3, "medium": 5, "deep": 7}[strategy]
        critical_layers = [0, self.num_layers//2, self.num_layers-1]
        active_layers = torch.argsort(layer_scores, descending=True)[:num_active].tolist()
        for layer in critical_layers:
            if layer not in active_layers:
                active_layers.append(layer)
        logger.info(f"Selected layers: {active_layers} for {strategy} strategy")
        self.selected_layers = active_layers  # Сохраняем для forward
        return active_layers

    def dynamic_threshold(self, relevance: torch.Tensor, layer_idx: int) -> float:
        # Flatten the relevance tensor to 1D
        flat_rel = relevance.view(-1)
        sorted_rel = torch.sort(flat_rel, descending=True)[0]
        if len(sorted_rel) == 0:
            return 0.0
        base_percent = 0.3 + (layer_idx / self.num_layers) * 0.2
        threshold_index = min(int(len(sorted_rel) * base_percent), len(sorted_rel) - 1)
        return sorted_rel[threshold_index].item()

    def _process_layer_with_rooms(self, hidden_states: torch.Tensor, layer_idx: int, threshold: float) -> torch.Tensor:
        active_rooms = self.select_rooms(hidden_states)
        attn_output, attn_weights = self.apply_attention(hidden_states, layer_idx, active_rooms)
        
        token_scores = attn_weights.sum(dim=(0, 1, 2))
        logger.info(f"Token scores: {token_scores}")
        active_tokens = self.token_tracker.update(token_scores, layer_idx, parent_stream_idx=layer_idx % self.token_tracker.max_streams)
        # active_tokens = self.token_tracker.update(token_scores, parent_stream_idx=layer_idx % self.token_tracker.max_streams)
        logger.info(f"Active tokens: {active_tokens}")
        
        return attn_output

    def apply_attention(self, hidden_states: torch.Tensor, layer_idx: int, active_rooms: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Загрузка блоков проекций
        q_block = self._load_block_with_cache(f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_q_proj_weight_block0")
        k_block = self._load_block_with_cache(f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_k_proj_weight_block0")
        v_block = self._load_block_with_cache(f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_v_proj_weight_block0")
        o_block = self._load_block_with_cache(f"{self.dispatcher.model_name}_layer{layer_idx}_self_attn_o_proj_weight_block0")
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_q_heads = 12  # Количество голов для query
        num_kv_heads = 2  # Количество голов для key/value
        head_dim = hidden_size // num_q_heads  # 128
        
        if torch.isnan(hidden_states).any():
            logger.error(f"NaN in hidden_states before layer {layer_idx}: {hidden_states[:, :5, :5]}")
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)
        
        # Проекции
        q = torch.matmul(hidden_states, q_block.t())
        k = torch.matmul(hidden_states, k_block.t())
        v = torch.matmul(hidden_states, v_block.t())
        
        q = q.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
        head_dim_kv = 256 // num_kv_heads
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim_kv).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim_kv).transpose(1, 2)
        
        # Rotary Positional Encoding
        positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        theta = 10000 ** (-2 * (torch.arange(head_dim // 2, device=hidden_states.device) / head_dim))
        angles = positions.unsqueeze(-1) * theta.unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(1).to(hidden_states.dtype)
        sin = torch.sin(angles).unsqueeze(1).to(hidden_states.dtype)
        
        q_0, q_1 = q[..., :head_dim//2], q[..., head_dim//2:]
        q = torch.cat([q_0 * cos - q_1 * sin, q_0 * sin + q_1 * cos], dim=-1)
        k_0, k_1 = k[..., :head_dim_kv//2], k[..., head_dim_kv//2:]
        k = torch.cat([k_0 * cos - k_1 * sin, k_0 * sin + k_1 * cos], dim=-1)
        
        k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        
        # Расчёт attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Causal маска
        mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        
        # Динамическая фильтрация токенов на основе значимости
        relevance_mask = self.token_tracker.get_relevance_mask(seq_len).to(self.device)
        attn_scores = attn_scores.masked_fill(~relevance_mask.unsqueeze(0).unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # Динамический порог для "горячих" токенов
        temp_threshold = self.dynamic_threshold(attn_scores, layer_idx)
        hot_mask = attn_scores > temp_threshold
        attn_scores = attn_scores.masked_fill(~hot_mask, float('-inf'))
        
        if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
            logger.error(f"NaN or Inf in attn_scores layer {layer_idx}: {attn_scores[:, :, :5, :5]}")
            attn_scores = torch.nan_to_num(attn_scores, nan=0.0, posinf=10.0, neginf=-10.0)
        
        logger.info(f"Attention scores shape: {attn_scores.shape}, values: {attn_scores[:, :, :5, :5]}")
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1).to(hidden_states.dtype)
        if torch.isnan(attn_weights).any():
            logger.error(f"NaN in attn_weights layer {layer_idx}: {attn_weights[:, :, :5, :5]}")
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        logger.info(f"Attention weights shape: {attn_weights.shape}, values: {attn_weights[:, :, :5, :5]}")
        
        # Обновляем значимость токенов на основе attention weights
        token_scores = attn_weights.sum(dim=(0, 1, 2))  # Суммируем веса внимания по всем головам и батчам
        self.token_tracker.update(token_scores, layer_idx, parent_stream_idx=layer_idx % self.token_tracker.max_streams)
        
        # "Шпионы": оцениваем релевантность токенов
        relevance_scores = self._compute_relevance(hidden_states, attn_weights)
        logger.info(f"Relevance scores: {relevance_scores}")
        
        # Применяем attention weights к values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = torch.matmul(attn_output, o_block.t())
        
        return attn_output, attn_weights
    
    def _compute_relevance(self, hidden_states: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет релевантность каждого токена на основе hidden states и attention weights.

        :param hidden_states: Скрытые состояния (shape: [batch_size, seq_len, hidden_size]).
        :param attn_weights: Веса внимания (shape: [batch_size, num_heads, seq_len, seq_len]).
        :return: Тензор релевантности (shape: [batch_size, seq_len]).
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        # Средние веса внимания для каждого токена
        avg_attn_weights = attn_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
        token_attn_scores = avg_attn_weights.sum(dim=1)  # [seq_len]
        
        # Конкатенируем hidden states и attention scores для оценки релевантности
        token_features = torch.cat([hidden_states, token_attn_scores.view(batch_size, seq_len, 1)], dim=-1)
        relevance_scores = self.relevance_scorer(token_features).sigmoid()  # [batch_size, seq_len, 1]
        return relevance_scores.squeeze(-1)  # [batch_size, seq_len]


    def forward(self, input_ids: torch.Tensor, temperature: float = 0.6, max_length: int = 50) -> Tuple[torch.Tensor, float]:
        batch_size = input_ids.shape[0]
        hidden_states = self._embed_input(input_ids)
        generated_ids = input_ids.clone()
        confidence = 0.0
        
        strategy = self.classify_query(hidden_states)
        self.select_layers(hidden_states, strategy)
        logger.info(f"Selected layers: {self.selected_layers}")
        
        eos_token_id = self.tokenizer.eos_token_id or 2
        max_steps = min(max_length - input_ids.shape[1], 50)
        
        for step in range(max_steps):
            next_hidden = hidden_states
            for layer_idx in self.selected_layers:
                next_hidden, attn_weights = self.apply_attention(next_hidden, layer_idx, active_rooms=[0])
                next_hidden = self._process_layer_with_rooms(next_hidden, layer_idx, threshold=0.1)
            
            if torch.isnan(next_hidden).any():
                logger.error(f"NaN in next_hidden step {step}: {next_hidden[:, -1:, :10]}")
                next_hidden = torch.nan_to_num(next_hidden, nan=0.0)
            
            logits = self._calculate_logits(next_hidden[:, -1:, :])
            if torch.all(logits == 0):
                logger.warning("Logits are all zeros, adding noise")
                logits = logits + torch.randn_like(logits) * 0.1
            
            # Динамическая температура
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
            dynamic_temperature = temperature * (1.0 + entropy)  # Увеличиваем температуру при высокой энтропии
            logits = logits / dynamic_temperature
            
            # Штрафы за повторение
            repeat_penalty = 1.2
            for token_id in generated_ids[0]:
                if token_id < logits.shape[-1]:
                    logits[0, 0, token_id] /= repeat_penalty
            
            # Усиливаем влияние ключевых токенов
            active_tokens = self.token_tracker.get_active_tokens()
            for token_id in range(logits.shape[-1]):
                if token_id not in active_tokens:
                    logits[0, 0, token_id] *= 0.5  # Уменьшаем вероятность нерелевантных токенов
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs[0, 0], num_samples=1)
            confidence += probs[0, 0, next_token].item()
            
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            hidden_states = self._embed_input(generated_ids)
            
            token_str = self.tokenizer.decode([next_token.item()])
            logger.info(f"Step {step}: Logits: {logits[0, 0, :10]}, Probs: {probs[0, 0, :10]}, Token '{token_str}' (ID: {next_token.item()}), Confidence: {confidence:.4f}, Temp: {dynamic_temperature:.2f}")
            
            if next_token.item() == eos_token_id:
                break
        
        return generated_ids, confidence
    
    def _calculate_logits(self, hidden_states: torch.Tensor, output_blocks: List[str] = None) -> torch.Tensor:
        if output_blocks is None:
            output_blocks = self.dispatcher.get_output_blocks(top_k=5)
        
        last_hidden = hidden_states[:, -1:, :]
        logger.info(f"Last hidden shape: {last_hidden.shape}")
        logger.info(f"Last hidden values: {last_hidden}")
        
        full_logits = []
        for block_key in output_blocks:
            block = self._load_block_with_cache(block_key)
            if block is None:
                logger.warning(f"Блок {block_key} не загружен")
                continue
            block_logits = torch.matmul(last_hidden.to(block.dtype), block.t())
            logger.info(f"Block {block_key} logits: {block_logits}")
            full_logits.append(block_logits)
        
        if not full_logits:
            logger.error("Не удалось загрузить ни один блок для вычисления логитов")
            return torch.zeros(hidden_states.shape[0], 1, self.vocab_size, device=self.device)
        
        logits = torch.cat(full_logits, dim=-1)
        logger.info(f"Logits shape: {logits.shape}")
        return logits

    def _load_block_with_cache(self, block_key: str) -> torch.Tensor:
        if block_key not in self.block_cache:
            self.block_cache[block_key] = self.dispatcher.load_block(block_key)
            logger.info(f"Loaded block {block_key} with shape {self.block_cache[block_key].shape}")
        return self.block_cache[block_key]

    def analyze_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.float16)
        hidden_states = F.layer_norm(hidden_states, normalized_shape=[hidden_states.shape[-1]])
        scores = self.attn_scorer(hidden_states).sigmoid()
        if torch.isnan(scores).any():
            logger.warning("Обнаружены NaN в attention scores!")
            scores = torch.nan_to_num(scores, nan=0.0)
        return hidden_states * scores

    def select_rooms(self, hidden_states: torch.Tensor) -> List[int]:
        scores = self.room_scorer(hidden_states.mean(dim=1)).sigmoid().squeeze(0)
        num_rooms = self.hidden_size // 256
        active_rooms = torch.argsort(scores, descending=True)[:self.top_k].tolist()
        return list(set(active_rooms + [0, 1]))

    def clear_memory(self):
        self.block_cache.clear()
        self.dispatcher.virtual_matrix.clear_cache()
        gc.collect()
        logger.info(f"Memory cleared, RAM: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    def reset(self):
        self.token_tracker = TokenTracker(self.vocab_size, max_streams=10, temp_threshold=0.6)
        self.block_cache.clear()
        self.clear_memory()
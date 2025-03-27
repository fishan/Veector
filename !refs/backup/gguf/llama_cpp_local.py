import ctypes
from ctypes import CFUNCTYPE, c_char_p, c_void_p, POINTER, Structure, c_int, c_float, c_int8
import logging
import os
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("Loading libllama.so")
lib_path = "/workspaces/Veector/llama.cpp/build/bin/libllama.so"
logger.debug(f"Library path: {lib_path}, modified: {os.path.getmtime(lib_path)}")
_lib = ctypes.CDLL(lib_path)
logger.debug("libllama.so loaded successfully")

class VirtualDispatcher(Structure):
    _fields_ = [
        ("gguf_path", c_char_p),
        ("load_block", CFUNCTYPE(c_void_p, c_char_p, c_void_p)),
        ("user_data", c_void_p)
    ]

class llama_model_params(Structure):
    _fields_ = [
        ("n_gpu_layers", c_int),
        ("split_mode", c_int),
        ("main_gpu", c_int),
        ("devices", POINTER(c_void_p)),
        ("use_mmap", c_int),
        ("check_tensors", c_int),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_int),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p)
    ]

class llama_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_int),
        ("n_batch", c_int),
        ("n_threads", c_int),
        ("n_threads_batch", c_int),
        ("rope_scaling_type", c_int),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_int),
        ("cb_eval", c_void_p),
        ("cb_eval_user_data", c_void_p),
        ("type_k", c_int),
        ("type_v", c_int),
        ("logits_all", c_int),
        ("embeddings", c_int),
        ("offload_kqv", c_int),
        ("abort_callback", c_void_p),
        ("abort_callback_data", c_void_p)
    ]

class llama_batch(Structure):
    _fields_ = [
        ("n_tokens", c_int),
        ("tokens", POINTER(c_int)),
        ("embd", POINTER(c_float)),
        ("pos", POINTER(c_int)),
        ("n_seq_id", POINTER(c_int)),
        ("seq_id", POINTER(POINTER(c_int))),
        ("logits", POINTER(c_int8)),
        ("all_pos_0", c_int),
        ("all_pos_1", c_int),
        ("all_seq_id", c_int)
    ]

# Определяем функции из llama.h
_lib.llama_model_load_from_file.argtypes = [c_char_p, POINTER(llama_model_params)]
_lib.llama_model_load_from_file.restype = c_void_p
_lib.llama_set_virtual_dispatcher.argtypes = [POINTER(VirtualDispatcher)]
_lib.llama_new_context_with_model.argtypes = [c_void_p, llama_context_params]
_lib.llama_new_context_with_model.restype = c_void_p
_lib.llama_tokenize.argtypes = [c_void_p, c_char_p, c_int, POINTER(c_int), c_int, c_int, c_int]
_lib.llama_tokenize.restype = c_int
_lib.llama_n_vocab.argtypes = [c_void_p]
_lib.llama_n_vocab.restype = c_int
_lib.llama_batch_init.argtypes = [c_int, c_int, c_int]
_lib.llama_batch_init.restype = llama_batch
_lib.llama_batch_free.argtypes = [llama_batch]
_lib.llama_decode.argtypes = [c_void_p, llama_batch]
_lib.llama_decode.restype = c_int
_lib.llama_get_logits.argtypes = [c_void_p]
_lib.llama_get_logits.restype = POINTER(c_float)
_lib.llama_token_to_piece.argtypes = [c_void_p, c_int, c_char_p, c_int]
_lib.llama_token_to_piece.restype = c_int
_lib.llama_free.argtypes = [c_void_p]
_lib.llama_free_model.argtypes = [c_void_p]

class Llama:
    def __init__(self, model_path, n_threads=4, n_ctx=2048, n_batch=512):
        logger.debug(f"ENTERING Llama.__init__ with model_path: {model_path}, n_threads: {n_threads}, n_ctx: {n_ctx}, n_batch: {n_batch}")
        params = llama_model_params(
            n_gpu_layers=0,
            split_mode=0,
            main_gpu=0,
            devices=ctypes.cast(None, POINTER(c_void_p)),
            use_mmap=1,
            check_tensors=0,
            kv_overrides=None,
            vocab_only=c_int(0),
            progress_callback=None,
            progress_callback_user_data=None
        )
        logger.debug(f"params: n_gpu_layers={params.n_gpu_layers}, split_mode={params.split_mode}, main_gpu={params.main_gpu}, devices={params.devices}, use_mmap={params.use_mmap}, check_tensors={params.check_tensors}, vocab_only={params.vocab_only}")
        logger.debug("Calling llama_model_load_from_file")
        self.model = _lib.llama_model_load_from_file(model_path.encode("utf-8"), ctypes.byref(params))
        if not self.model:
            logger.error("Failed to load llama model")
            raise RuntimeError("Failed to load llama model")
        
        ctx_params = llama_context_params(
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            rope_scaling_type=-1,
            rope_freq_base=0.0,
            rope_freq_scale=0.0,
            yarn_ext_factor=-1.0,
            yarn_attn_factor=1.0,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            yarn_orig_ctx=0,
            cb_eval=None,
            cb_eval_user_data=None,
            type_k=1,  # GGML_TYPE_F16
            type_v=1,  # GGML_TYPE_F16
            logits_all=0,
            embeddings=0,
            offload_kqv=0,
            abort_callback=None,
            abort_callback_data=None
        )
        self.ctx = _lib.llama_new_context_with_model(self.model, ctx_params)
        if not self.ctx:
            logger.error("Failed to create llama context")
            raise RuntimeError("Failed to create llama context")
        
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.dispatcher = None  # Сохраняем диспетчер для отладки
        logger.debug("Llama model and context loaded successfully")

    def set_virtual_dispatcher(self, dispatcher):
        logger.debug("Setting virtual dispatcher")
        @CFUNCTYPE(c_void_p, c_char_p, c_void_p)
        def load_block(block_key, user_data):
            key = block_key.decode("utf-8")
            logger.debug(f"Calling load_block for key: {key}")
            block = dispatcher.load_block(key)
            if block is None:
                logger.error(f"Failed to load block: {key}")
                return None
            # Убеждаемся, что данные остаются в памяти
            result = block.ctypes.data_as(c_void_p)
            logger.debug(f"load_block result: {result}, shape: {block.shape}, size: {block.nbytes}")
            return result
        
        self.dispatcher = dispatcher  # Сохраняем для отладки
        c_dispatcher = VirtualDispatcher()
        gguf_path = getattr(dispatcher, "gguf_path", "")
        c_dispatcher.gguf_path = gguf_path.encode("utf-8")
        c_dispatcher.load_block = load_block
        c_dispatcher.user_data = None
        logger.debug("Calling llama_set_virtual_dispatcher")
        _lib.llama_set_virtual_dispatcher(ctypes.byref(c_dispatcher))
        logger.debug("Virtual dispatcher set successfully")

    def __call__(self, input_data, max_tokens=200):
        logger.debug(f"Calling Llama with input: {input_data}")
        
        # Токенизация входных данных
        if isinstance(input_data, str):
            input_str = input_data.encode("utf-8")
            tokens = (c_int * self.n_ctx)()
            n_tokens = _lib.llama_tokenize(self.model, input_str, len(input_str), tokens, self.n_ctx, True, False)
            if n_tokens < 0:
                logger.error("Tokenization failed")
                return {"choices": [{"text": "Error: Tokenization failed"}]}
            input_tokens = [tokens[i] for i in range(n_tokens)]
        else:
            input_tokens = list(input_data)
            n_tokens = len(input_tokens)
            if n_tokens > self.n_ctx:
                logger.warning(f"Input tokens exceed context size: {n_tokens} > {self.n_ctx}, truncating")
                input_tokens = input_tokens[:self.n_ctx]
                n_tokens = self.n_ctx

        # Инициализация батча для входных токенов
        batch = _lib.llama_batch_init(max(n_tokens, 1), 0, 1)
        token_array = (c_int * n_tokens)(*input_tokens)
        pos_array = (c_int * n_tokens)(*range(n_tokens))
        seq_id_array = (c_int * n_tokens)(*[0] * n_tokens)
        logits_array = (c_int8 * n_tokens)(*[0] * (n_tokens - 1) + [1])  # Логиты для последнего токена

        batch.n_tokens = n_tokens
        batch.tokens = ctypes.cast(token_array, POINTER(c_int))
        batch.pos = ctypes.cast(pos_array, POINTER(c_int))
        batch.n_seq_id = ctypes.cast((c_int * n_tokens)(*[1] * n_tokens), POINTER(c_int))
        batch.seq_id = ctypes.cast((POINTER(c_int) * n_tokens)(*[ctypes.cast(ctypes.byref(seq_id_array[i]), POINTER(c_int)) for i in range(n_tokens)]), POINTER(POINTER(c_int)))
        batch.logits = ctypes.cast(logits_array, POINTER(c_int8))
        batch.all_pos_0 = 0
        batch.all_pos_1 = 1
        batch.all_seq_id = 0

        # Декодируем входные токены
        logger.debug(f"Decoding input batch: n_tokens={batch.n_tokens}")
        if _lib.llama_decode(self.ctx, batch) != 0:
            logger.error("Failed to decode input tokens")
            _lib.llama_batch_free(batch)
            return {"choices": [{"text": "Error: Decode failed"}]}

        # Генерация новых токенов
        output_tokens = input_tokens.copy()
        n_vocab = _lib.llama_n_vocab(self.model)
        batch_gen = _lib.llama_batch_init(1, 0, 1)  # Батч для генерации
        
        for _ in range(max_tokens):
            logits = _lib.llama_get_logits(self.ctx)
            if not logits:
                logger.error("Failed to get logits")
                break
            
            logits_array = [logits[i] for i in range(n_vocab)]
            next_token = np.argmax(logits_array)
            output_tokens.append(next_token)

            # Подготовка батча для следующего токена
            token_gen = (c_int * 1)(next_token)
            pos_gen = (c_int * 1)(len(output_tokens) - 1)
            seq_id_gen = (c_int * 1)(0)
            logits_gen = (c_int8 * 1)(1)
            
            batch_gen.n_tokens = 1
            batch_gen.tokens = ctypes.cast(token_gen, POINTER(c_int))
            batch_gen.pos = ctypes.cast(pos_gen, POINTER(c_int))
            batch_gen.n_seq_id = ctypes.cast((c_int * 1)(1), POINTER(c_int))
            batch_gen.seq_id = ctypes.cast((POINTER(c_int) * 1)(ctypes.cast(ctypes.byref(seq_id_gen[0]), POINTER(c_int))), POINTER(POINTER(c_int)))
            batch_gen.logits = ctypes.cast(logits_gen, POINTER(c_int8))
            batch_gen.all_pos_0 = 0
            batch_gen.all_pos_1 = 1
            batch_gen.all_seq_id = 0

            logger.debug(f"Decoding generated token: {next_token}")
            if _lib.llama_decode(self.ctx, batch_gen) != 0:
                logger.error("Failed to decode generated token")
                break

            if next_token == 151643:  # EOS для Qwen2
                break

        # Преобразуем токены в текст
        output_text = ""
        buf = (c_char * 256)()
        for token in output_tokens:
            n_chars = _lib.llama_token_to_piece(self.model, token, buf, 256)
            if n_chars > 0:
                output_text += buf[:n_chars].decode("utf-8", errors="ignore")

        _lib.llama_batch_free(batch)
        _lib.llama_batch_free(batch_gen)
        return {"choices": [{"text": output_text}]}

    def __del__(self):
        logger.debug("Destroying Llama instance")
        if hasattr(self, 'ctx') and self.ctx:
            _lib.llama_free(self.ctx)
        if hasattr(self, 'model') and self.model:
            _lib.llama_free_model(self.model)
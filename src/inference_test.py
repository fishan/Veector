# FILE: qwen_inference.py
# Description: Script to run inference using Veector core and converted model tensors.
# Version: 0.2.6 (Added prompt template, sampling params, fixed group ID)

import numpy as np
import time
import pickle
import os
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Version ---
QWEN_INFERENCE_VERSION = "0.2.6" # Added prompt template, sampling params, fixed group ID

# --- Inference & Sampling Parameters (from GGUF / Recommendations) ---
# <<< ДОБАВЛЕНО >>> Параметры сэмплирования (не используются в текущей argmax логике)
TEMPERATURE = 0.6
TOP_K = 20
TOP_P = 0.9
MIN_P = 0.1
# N_CTX = 4096 # Not directly used in this script
# N_BATCH = 512 # Not directly used in this script

# <<< ДОБАВЛЕНО >>> Шаблон промпта
PROMPT_TEMPLATE = "<｜User｜>{message}<｜Assistant｜>"

# --- Необходимые импорты из проекта ---
PROJECT_IMPORTS_OK = False
CORE_VERSION_REQ = "0.6.3"
TENSORS_VERSION_REQ = "0.7.6"
VEECTORDB_VERSION_REQ = "0.9.7" # Обновлено для полноты

try:
    from core import Veector, CORE_VERSION
    print(f"  Imported Core (v{CORE_VERSION})")
    if CORE_VERSION < CORE_VERSION_REQ:
        raise ImportError(f"qwen_inference.py requires core v{CORE_VERSION_REQ}+, found v{CORE_VERSION}")

    from tensors import TensorCoordinate, TENSORS_VERSION
    # Импортируем ОБЕ константы, чтобы выбрать правильную
    from tensors import GROUP_IDX_QWEN_KNOWLEDGE, GROUP_IDX_DEEPSEEK_KNOWLEDGE
    print(f"  Imported Tensors (v{TENSORS_VERSION})")
    if TENSORS_VERSION < TENSORS_VERSION_REQ:
        raise ImportError(f"qwen_inference.py requires tensors v{TENSORS_VERSION_REQ}+, found v{TENSORS_VERSION}")

    from transformers import AutoTokenizer, PreTrainedTokenizer
    print("Project components imported successfully.")
    PROJECT_IMPORTS_OK = True

# --- Обработка ошибок импорта ---
except ImportError as e:
    print(f"---!!! FATAL ERROR (ImportError in qwen_inference.py) !!! ---")
    print(f"Specific error: {e}")
    print(f"Ensure files (core v{CORE_VERSION_REQ}+, tensors v{TENSORS_VERSION_REQ}+) are OK.")
    print(f"-----------------------------------------")
    PROJECT_IMPORTS_OK = False
except NameError as ne:
    print(f"---!!! FATAL ERROR (NameError in qwen_inference.py) !!! ---")
    print(f"Specific error: {ne}")
    print(f"-----------------------------------------")
    PROJECT_IMPORTS_OK = False
except Exception as import_e:
    print(f"---!!! FATAL ERROR (Other Exception during Import in qwen_inference.py) !!! ---")
    print(f"Specific error: {import_e}")
    traceback.print_exc()
    print(f"----------------------------------------------------------")
    PROJECT_IMPORTS_OK = False


def run_qwen_inference(
    text: str,
    db_path: Union[str, Path],
    model_name_hf: str,
    nest_level: int = 1,
    num_layers: int = 28,
    # <<< ДОБАВЛЕНО >>> Опциональные параметры сэмплирования для будущей реализации
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    min_p: float = MIN_P
    ):
    """
    Запускает инференс (предсказание 1 токена).
    Включает форматирование промпта.
    Параметры сэмплирования (temp, top_k/p, min_p) пока не используются.
    """
    print(f"\n--- Running Inference Script v{QWEN_INFERENCE_VERSION} ---")
    if not PROJECT_IMPORTS_OK: print("Cannot run: failed imports."); return

    # --- Логирование параметров ---
    print(f"Original Input Text: '{text}'")
    # <<< ДОБАВЛЕНО >>> Применяем шаблон промпта
    formatted_text = PROMPT_TEMPLATE.format(message=text)
    print(f"Formatted Input Text: '{formatted_text}'")
    db_path = Path(db_path)
    print(f"DB Path: {db_path.resolve()}"); print(f"HF Model Name: {model_name_hf}")
    print(f"Target Nest Level: {nest_level}"); print(f"Number of Layers: {num_layers}")
    print(f"Sampling Params (info only): Temp={temperature}, TopK={top_k}, TopP={top_p}, MinP={min_p}")
    if not db_path.is_dir(): print(f"ERROR: DB directory not found: {db_path}"); return

    # --- Загрузка Токенизатора ---
    tokenizer: Optional[PreTrainedTokenizer] = None
    try:
        print(f"Loading tokenizer '{model_name_hf}'..."); tokenizer_path = f"deepseek-ai/{model_name_hf}"
        # <<< ИЗМЕНЕНО >>> Добавлен use_fast=False, если возникают проблемы с шаблоном/спецтокенами
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        print(f"Tokenizer loaded from {tokenizer_path}.")
        if tokenizer.pad_token is None:
            # Проверяем наличие стандартных токенов перед присвоением eos_token
            if tokenizer.eos_token:
                print("Tokenizer missing pad_token, setting to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Если нет и eos_token, добавляем простой pad_token
                print("Tokenizer missing pad_token and eos_token. Adding '<pad>'.")
                tokenizer.add_special_tokens({'pad_token': '<pad>'})

    except Exception as e: print(f"ERROR: Failed tokenizer load: {e}"); traceback.print_exc(); return

    # --- Загрузка Карты Процессоров ---
    processor_map: Dict[str, str] = {}
    proc_map_file = db_path / f"{model_name_hf}_proc_map.pkl"
    if proc_map_file.is_file():
        try:
            with open(proc_map_file, 'rb') as f: processor_map = pickle.load(f)
            print(f"Loaded processor map ({len(processor_map)} entries) from {proc_map_file}")
        except Exception as e: print(f"Warning: Failed processor map load: {e}.")
    else: print(f"ERROR: Processor map file not found: {proc_map_file}"); return

    # --- Инициализация Veector ---
    vec: Optional[Veector] = None
    try:
        vec = Veector(db_dir=db_path)
        print(f"Veector core v{CORE_VERSION} initialized using DB at: {vec.db.db_root_path.resolve()}")
    except Exception as e: print(f"FATAL: Veector init failed: {e}"); return

    # --- Проверка Необходимых Процессоров ---
    required_proc_keys = ["embedding", "final_norm", "lm_head"]
    for i in range(num_layers): required_proc_keys.extend([f"attn_{i}", f"ffn_{i}"])
    missing_procs = [key for key in required_proc_keys if key not in processor_map]
    if missing_procs: print(f"ERROR: Required processors missing: {missing_procs}"); vec.db.close(); return
    embedding_processor_id = processor_map["embedding"]
    final_norm_id = processor_map["final_norm"]
    lm_head_id = processor_map["lm_head"]
    print("All required processor IDs found in map.")

    # --- Подготовка Входных Данных ---
    input_ids: Optional[np.ndarray] = None; attention_mask: Optional[np.ndarray] = None; position_ids: Optional[np.ndarray] = None
    if tokenizer:
        try:
            # <<< ИЗМЕНЕНО >>> Токенизируем отформатированный текст
            # Важно: add_special_tokens=False может быть нужно, если шаблон уже содержит спецтокены начала/конца
            # Это зависит от того, как токенизатор обрабатывает шаблон. Попробуем с True по умолчанию.
            inputs = tokenizer(formatted_text, return_tensors="np", padding=False, truncation=False, add_special_tokens=True)
            input_ids = inputs['input_ids']; attention_mask = inputs.get('attention_mask')
            seq_length = input_ids.shape[1]; position_ids = np.arange(0, seq_length, dtype=np.int64).reshape(1, -1)
            if attention_mask is None: attention_mask = np.ones_like(input_ids)
            print(f"\n--- Prepared Input ---"); print(f"Input IDs shape: {input_ids.shape}"); print(f"Position IDs shape: {position_ids.shape}"); print(f"Attention Mask shape: {attention_mask.shape}")
            # Распечатаем ID для отладки
            print(f"Input IDs: {input_ids[0].tolist()}")
            print(f"Decoded Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
        except Exception as e: print(f"Error tokenizing: {e}"); traceback.print_exc(); vec.db.close(); return
    if input_ids is None: print("ERROR: input_ids are None."); vec.db.close(); return

    # --- Запуск Инференса ---
    current_hidden_states: Optional[np.ndarray] = None; final_logits: Optional[np.ndarray] = None
    start_inference_time = time.time()
    # Используем ID группы знаний = 100 (как в конвертере)
    knowledge_group_id = GROUP_IDX_QWEN_KNOWLEDGE # ID = 100
    print(f"\n--- Starting Inference (using knowledge group {knowledge_group_id}) ---")
    try:
        # 1. Embedding Layer
        print(f"Running Embedding Processor ({embedding_processor_id})...")
        compute_context_embed = { "input_data": input_ids, "required_nest": nest_level, "target_knowledge_group": knowledge_group_id }
        embed_result = vec.compute(embedding_processor_id, context=compute_context_embed)
        if not (embed_result and embed_result.get("status") == "completed"): raise RuntimeError(f"Embedding failed: {embed_result}")
        current_hidden_states = embed_result.get("data")
        if current_hidden_states is None: raise RuntimeError("Embedding returned None data.")
        print(f"  Embed Output Shape: {current_hidden_states.shape}, Dtype: {current_hidden_states.dtype}")

        # --- Цикл по Слоям Трансформера ---
        print("\n--- Running Transformer Layers ---")
        residual_input = current_hidden_states

        for layer_idx in range(num_layers):
            layer_start_time = time.time()
            print(f"\n--- Processing Layer {layer_idx}/{num_layers-1} ---")
            attn_proc_id = processor_map[f"attn_{layer_idx}"]
            ffn_proc_id = processor_map[f"ffn_{layer_idx}"]

            # --- Attention Block ---
            print(f"  Running Attn Layer {layer_idx} ({attn_proc_id})...")
            attn_context = { "input_data": current_hidden_states, "residual_input": residual_input,
                             "required_nest": nest_level, "target_knowledge_group": knowledge_group_id,
                             "position_ids": position_ids, "attention_mask": attention_mask }
            attn_result = vec.compute(attn_proc_id, context=attn_context)
            if not (attn_result and attn_result.get("status") == "completed"): raise RuntimeError(f"Attn L{layer_idx} failed: {attn_result}")
            attn_output = attn_result.get("data");
            if attn_output is None: raise RuntimeError(f"Attn L{layer_idx} returned None data.")
            print(f"  Attn L{layer_idx} Output Shape: {attn_output.shape}, Dtype: {attn_output.dtype}")
            current_hidden_states = attn_output; residual_input_ffn = attn_output

            # --- FFN Block ---
            print(f"\n  Running FFN Layer {layer_idx} ({ffn_proc_id})...")
            ffn_context = { "input_data": current_hidden_states, "residual_input": residual_input_ffn,
                            "required_nest": nest_level, "target_knowledge_group": knowledge_group_id }
            ffn_result = vec.compute(ffn_proc_id, context=ffn_context)
            if not (ffn_result and ffn_result.get("status") == "completed"): raise RuntimeError(f"FFN L{layer_idx} failed: {ffn_result}")
            ffn_output = ffn_result.get("data");
            if ffn_output is None: raise RuntimeError(f"FFN L{layer_idx} returned None data.")
            print(f"  FFN L{layer_idx} Output Shape: {ffn_output.shape}, Dtype: {ffn_output.dtype}")
            current_hidden_states = ffn_output; residual_input = ffn_output

            layer_end_time = time.time()
            print(f"--- Finished Layer {layer_idx} in {layer_end_time - layer_start_time:.3f} seconds ---")
        # --- Конец Цикла по Слоям ---

        # --- Финальная Нормализация ---
        print(f"\nRunning Final LayerNorm ({final_norm_id})...")
        norm_context = { "input_data": current_hidden_states, "required_nest": nest_level, "target_knowledge_group": knowledge_group_id }
        norm_result = vec.compute(final_norm_id, context=norm_context)
        if not (norm_result and norm_result.get("status") == "completed"): raise RuntimeError(f"Final Norm failed: {norm_result}")
        final_normed_states = norm_result.get("data")
        if final_normed_states is None: raise RuntimeError("Final Norm returned None data.")
        print(f"Final Norm Output Shape: {final_normed_states.shape}, Dtype: {final_normed_states.dtype}")

        # --- LM Head ---
        print(f"\nRunning LM Head ({lm_head_id})...")
        lm_head_context = { "input_data": final_normed_states, "required_nest": nest_level, "target_knowledge_group": knowledge_group_id }
        logits_result = vec.compute(lm_head_id, context=lm_head_context)
        if not (logits_result and logits_result.get("status") == "completed"): raise RuntimeError(f"LM Head failed: {logits_result}")
        final_logits = logits_result.get("data")
        if final_logits is None: raise RuntimeError("LM Head returned None data.")
        print(f"Final Logits Shape: {final_logits.shape}, Dtype: {final_logits.dtype}")

        # --- Получение Предсказанного Токена ---
        print("\n--- Prediction ---")
        if tokenizer and final_logits is not None:
             last_token_logits = final_logits[0, -1, :] # Логиты для последнего токена во входной последовательности

             # --- ЛОГИКА ВЫБОРА СЛЕДУЮЩЕГО ТОКЕНА ---
             # Текущая реализация: Просто выбираем самый вероятный (argmax)
             predicted_token_id = np.argmax(last_token_logits)
             predicted_token = tokenizer.decode(predicted_token_id, skip_special_tokens=True) # Добавлен skip_special_tokens

             print(f"Argmax Predicted next token ID: {predicted_token_id}")
             print(f"Argmax Predicted next token: '{predicted_token}'")

             # --- ЗАКОММЕНТИРОВАНАЯ ЛОГИКА СЭМПЛИРОВАНИЯ (требует цикла генерации) ---
             # # 1. Применить температуру
             # scaled_logits = last_token_logits / temperature
             # # 2. Применить Top-K / Top-P / Min-P (нужна функция)
             # filtered_logits = apply_sampling_filters(scaled_logits, top_k=top_k, top_p=top_p, min_p=min_p)
             # # 3. Получить вероятности через Softmax
             # probabilities = softmax(filtered_logits) # Нужна реализация softmax, если не импортирована
             # # 4. Сэмплировать ID токена на основе вероятностей
             # sampled_token_id = np.random.choice(len(probabilities), p=probabilities)
             # sampled_token = tokenizer.decode(sampled_token_id)
             # print(f"\nSampled next token ID: {sampled_token_id}")
             # print(f"Sampled next token: '{sampled_token}'")
             # --- КОНЕЦ ЗАКОММЕНТИРОВАННОЙ ЛОГИКИ ---

        elif not tokenizer: print("Cannot decode: tokenizer unavailable.")
        else: print("Cannot decode: final_logits are None.")

    except Exception as e: print(f"\n--- ERROR during inference execution ---"); print(f"{e}"); traceback.print_exc()
    finally:
        if vec and hasattr(vec, 'db') and vec.db: vec.db.close(); print("\nDatabase connection closed.")

    end_inference_time = time.time()
    print(f"\n--- Inference Script Finished in {end_inference_time - start_inference_time:.3f} seconds ---")

# --- Вспомогательная функция для сэмплирования (ЗАГЛУШКА) ---
def apply_sampling_filters(logits: np.ndarray, top_k: int, top_p: float, min_p: float) -> np.ndarray:
    """
    Применяет фильтры Top-K, Top-P, Min-P к логитам.
    (Требует реализации)
    """
    print("WARN: apply_sampling_filters not implemented, returning original logits.")
    # TODO: Implement filtering logic
    # 1. Top-K: Занулить логиты всех токенов, кроме K самых вероятных.
    # 2. Top-P: Отсортировать по убыванию вероятностей, выбрать топ токены, чья сумма вероятностей >= P.
    # 3. Min-P: Убрать токены с вероятностью < Min-P * max_probability.
    return logits

# --- Main Execution ---
if __name__ == "__main__":
    # Определяем пути относительно скрипта
    script_dir = Path(__file__).parent.resolve()
    target_db_path = script_dir.parent / "data" / "db"

    model_hf = "DeepSeek-R1-Distill-Qwen-1.5B"
    input_text = "The best way to predict the future is to"
    inference_nest = 1
    model_num_layers = 28

    print(f"--- Starting Inference ---"); print(f"DB: {target_db_path.resolve()}"); print(f"Input: '{input_text}'"); print(f"Nest: {inference_nest}")
    try: print(f"Core: {CORE_VERSION}")
    except NameError: print("Core version unknown")
    try: print(f"Tensors: {TENSORS_VERSION}")
    except NameError: print("Tensors version unknown")
    try:
         from veectordb import VEECTORDB_VERSION
         print(f"DB: {VEECTORDB_VERSION}")
    except (ImportError, NameError): print("DB version unknown")

    if target_db_path.is_dir():
         run_qwen_inference(
             text=input_text,
             db_path=target_db_path,
             model_name_hf=model_hf,
             nest_level=inference_nest,
             num_layers=model_num_layers,
             # Можно передать параметры сэмплирования, но они пока не используются
             # temperature=TEMPERATURE,
             # top_k=TOP_K,
             # top_p=TOP_P,
             # min_p=MIN_P
        )
    else: print(f"ERROR: Database directory not found at {target_db_path}")
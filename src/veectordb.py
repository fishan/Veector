# FILE: veectordb.py
# Version: 0.9.7 (Hybrid Approach: Handles metadata_extra in Tuple)

import os
import pickle
import zlib
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
import traceback

# --- Version ---
VEECTORDB_VERSION = "0.9.7" # Handles metadata_extra in Tuple

# --- Импорты из tensors.py (v0.7.6+) ---
# <<< ИЗМЕНЕНО >>> Обновляем требуемую версию tensors
TENSORS_VERSION_REQ = "0.7.6"
try:
    from tensors import TENSORS_VERSION
    if TENSORS_VERSION < TENSORS_VERSION_REQ:
         raise ImportError(f"VeectorDB v{VEECTORDB_VERSION} requires tensors.py v{TENSORS_VERSION_REQ}+, found v{TENSORS_VERSION}")

    # Импортируем ВСЕ необходимое и потенциально необходимое
    from tensors import (
        # Основное
        TensorCoordinate, MetadataTuple,
        validate_tensor, validate_tensor_tuple, get_tensor_hash,
        # Новые геттеры для кортежа (включая новый get_metadata_extra_from_meta)
        get_data_description_from_meta, get_version_from_meta, get_type_code_from_meta,
        get_dtype_code_from_meta, get_name_id_from_meta, get_has_blob_flag_from_meta,
        get_coord_list_from_meta, get_coord_obj_from_meta, get_shape_list_from_meta,
        get_tags_list_from_meta, get_ops_sequences_from_meta, get_interface_from_meta,
        get_filters_from_meta, get_exit_gates_from_meta, get_lifecycle_list_from_meta,
        get_status_code_from_meta, get_evo_version_from_meta, get_parents_list_from_meta,
        get_metadata_extra_from_meta, # <<< ДОБАВЛЕНО >>> Импорт нового геттера
        # Старые геттеры для списка (нужны для реконструкции)
        get_tensor_metadata, get_tensor_coord, get_tensor_type, get_tensor_status,
        get_tensor_tags, has_blob_data, get_tensor_parents, get_tensor_op_channels,
        get_tensor_filters, get_tensor_exit_gates,
        # Константы Тегов
        TAG_TYPE_PROCESSOR, TAG_TYPE_KNOWLEDGE, TAG_TYPE_CONVERTER, TAG_TYPE_STATE,
        TAG_MODEL_QWEN2, TAG_MODEL_LLAMA3, TAG_MODEL_DEEPSEEK,
        TAG_PREC_FLOAT32, TAG_PREC_FLOAT16, TAG_PREC_BFLOAT16, TAG_PREC_INT8, TAG_PREC_INT4,
        TAG_COMP_WEIGHTS, TAG_COMP_BIAS, TAG_COMP_EMBEDDING, TAG_COMP_ATTN_Q, TAG_COMP_ATTN_K,
        TAG_COMP_ATTN_V, TAG_COMP_ATTN_O, TAG_COMP_ATTN_QKV, TAG_COMP_FFN_GATE,
        TAG_COMP_FFN_UP, TAG_COMP_FFN_DOWN, TAG_COMP_LAYERNORM, TAG_COMP_LM_HEAD,
        # Маппинги
        DATA_TYPE_MAPPING, REVERSE_DATA_TYPE_MAPPING,
        DTYPE_MAPPING, REVERSE_DTYPE_MAPPING,
        STATUS_MAPPING, REVERSE_STATUS_MAPPING
    )
except ImportError as e:
    print(f"---!!! FATAL ERROR: Cannot import required components from tensors.py (v{TENSORS_VERSION_REQ}+) in veectordb.py: {e} !!!---")
    raise
except Exception as e_other:
    print(f"---!!! FATAL ERROR: Unexpected error importing from tensors.py in veectordb.py: {e_other} !!!---")
    traceback.print_exc()
    raise

class VeectorDB:
    """
    Veector Database V0.9.7
    - insert/hash/index работают с КОРТЕЖЕМ метаданных (11 элементов).
    - get возвращает СТРУКТУРУ СПИСКА, корректно включая metadata_extra.
    """
    INDEX_FILENAME = "tensor_index.pkl"

    def __init__(self, db_dir: Union[str, Path] = "data/db"):
        # <<< ИЗМЕНЕНО >>> Обновлена версия в логе
        print(f"--- Initializing VeectorDB v{VEECTORDB_VERSION} (requires tensors v{TENSORS_VERSION_REQ}+) ---")
        self.db_root_path = Path(db_dir).resolve()
        self.index_path = self.db_root_path / self.INDEX_FILENAME
        self.index: Dict[str, Dict] = {}
        self._index_dirty = False
        self.db_root_path.mkdir(parents=True, exist_ok=True)
        self._load_index()
        print(f"VeectorDB v{VEECTORDB_VERSION} initialized at {self.db_root_path}. Index entries: {len(self.index)}.")

    # --- Загрузка/Сохранение Индекса ---
    # (Без изменений)
    def _load_index(self):
        if self.index_path.is_file():
            try:
                with open(self.index_path, 'rb') as f: self.index = pickle.load(f)
                if not isinstance(self.index, dict): self.index = {}
            except Exception as e: print(f"Warn: Load index failed: {e}"); self.index = {}
        else: self.index = {}
        self._index_dirty = False

    def _save_index(self):
        if self._index_dirty:
            current_index_size_before_save = len(self.index)
            print(f"DEBUG INDEX SAVE (v{VEECTORDB_VERSION}): Attempting DIRECT save.")
            print(f"DEBUG INDEX SAVE: Size in memory BEFORE save: {current_index_size_before_save}")
            try: last_keys = list(self.index.keys())[-5:]; print(f"DEBUG INDEX SAVE: Last 5 keys in memory: {last_keys}")
            except Exception: print("DEBUG INDEX SAVE: Could not get last keys.")
            try:
                with open(self.index_path, 'wb') as f: pickle.dump(self.index, f, pickle.HIGHEST_PROTOCOL)
                self._index_dirty = False
                print(f"DEBUG INDEX SAVE: pickle.dump completed for {self.index_path}.")
                print(f"DEBUG INDEX SAVE: Attempting immediate reload for verification...")
                reloaded_index = None; reloaded_size = 0
                try:
                     with open(self.index_path, 'rb') as f_verify: reloaded_index = pickle.load(f_verify)
                     if isinstance(reloaded_index, dict):
                          reloaded_size = len(reloaded_index)
                          print(f"DEBUG INDEX SAVE: Immediate reload SUCCESS. Size loaded: {reloaded_size}")
                          if reloaded_size != current_index_size_before_save: print(f"---!!! CRITICAL WARNING: Size mismatch after save! Expected {current_index_size_before_save}, Loaded {reloaded_size} !!!---")
                     else: print(f"---!!! CRITICAL WARNING: Reloaded index is not a dict! Type: {type(reloaded_index)} !!!---")
                except Exception as reload_e: print(f"---!!! CRITICAL WARNING: Failed to reload index immediately after saving: {reload_e} !!!---")
            except Exception as e:
                print(f"---!!! ERROR: Failed to DIRECTLY save index file {self.index_path}: {e} !!!---")
                try:
                    if self.index_path.exists(): self.index_path.unlink(); print(f"  Deleted potentially corrupted index file.")
                except OSError: pass
        current_index_size_after_save = len(self.index)
        print(f"DEBUG INDEX SAVE: Size in memory AFTER save: {current_index_size_after_save}")

    def close(self):
        print(f"Closing VeectorDB v{VEECTORDB_VERSION} connection..."); self._save_index(); self.index = {}
    def __del__(self):
        if hasattr(self, '_save_index') and callable(self._save_index): self._save_index()

    # --- Вспомогательные Методы для Путей ---
    # (Без изменений)
    def _get_structured_dir_path(self, coord: TensorCoordinate) -> Path:
        if not isinstance(coord, TensorCoordinate): raise TypeError("Invalid TensorCoordinate object")
        return self.db_root_path / f"g{coord.group}" / f"l{coord.layer}" / f"n{coord.nest}"
    def _get_meta_file_path(self, coord: TensorCoordinate, tensor_id: str) -> Path:
        dir_path = self._get_structured_dir_path(coord); return dir_path / f"{tensor_id}.meta"
    def _get_blob_file_path(self, coord: TensorCoordinate, tensor_id: str, blob_format: str) -> Path:
        dir_path = self._get_structured_dir_path(coord); suffix = ".npy" if blob_format == 'npy' else ".blob"; return dir_path / f"{tensor_id}{suffix}"

    # --- Методы Хранения/Загрузки Блобов ---
    # (Без изменений)
    def _store_pickle_blob(self, data: Any, blob_file_path: Path) -> bool:
        if data is None: return False
        try:
            serialized_data = pickle.dumps(data); compressed_data = zlib.compress(serialized_data)
            blob_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(blob_file_path, "wb") as f: f.write(compressed_data)
            return True
        except Exception as e: print(f"---!!! ERROR storing pickle blob {blob_file_path}: {e} !!!---"); return False
    def _store_npy_blob(self, data: np.ndarray, blob_file_path: Path) -> bool:
        if not isinstance(data, np.ndarray): print(f"Error: Expected numpy array, got {type(data)}."); return False
        try:
            blob_file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(blob_file_path, data, allow_pickle=False)
            return True
        except Exception as e: print(f"---!!! ERROR storing npy blob {blob_file_path}: {e} !!!---"); return False
    def _load_pickle_blob(self, blob_file_path: Path) -> Any | None:
        if not blob_file_path.is_file(): return None
        try:
            with open(blob_file_path, "rb") as f: compressed_data = f.read()
            if not compressed_data: return None
            serialized_data = zlib.decompress(compressed_data); del compressed_data
            data = pickle.loads(serialized_data); return data
        except Exception as e: print(f"---!!! ERROR loading pickle blob {blob_file_path}: {e} !!!---"); return None
    def _load_npy_blob(self, blob_file_path: Path, use_mmap: bool = True) -> Any | None:
        if not blob_file_path.is_file(): return None
        try:
            mmap_mode = 'r' if use_mmap else None
            data = np.load(blob_file_path, mmap_mode=mmap_mode, allow_pickle=False)
            return data
        except Exception as e: print(f"---!!! ERROR loading npy blob {blob_file_path}: {e} !!!---"); return None

    # --- Обработка Файлов Метаданных (Работаем с КОРТЕЖЕМ) ---
    # (Без изменений)
    def _load_meta_tuple_from_file(self, meta_file_path: Path) -> Optional[MetadataTuple]:
        if not meta_file_path.is_file(): return None
        try:
            with open(meta_file_path, 'rb') as f: meta_tuple = pickle.load(f)
            if validate_tensor_tuple(meta_tuple): return meta_tuple
            else: print(f"---!!! WARNING: Invalid metadata tuple loaded from {meta_file_path} !!!---"); return None
        except Exception as e: print(f"---!!! ERROR loading meta tuple file {meta_file_path}: {e} !!!---"); return None
    def _save_meta_tuple_to_file(self, meta_tuple: MetadataTuple, meta_file_path: Path) -> bool:
        try:
            if not validate_tensor_tuple(meta_tuple):
                 print(f"---!!! ERROR: Attempting to save invalid metadata tuple to {meta_file_path} !!!---"); return False
            meta_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_file_path, 'wb') as f: pickle.dump(meta_tuple, f, pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e: print(f"---!!! ERROR writing meta tuple file {meta_file_path}: {e} !!!---"); return False

    # --- Управление Индексом (Работа с КОРТЕЖЕМ) ---
    # (Без изменений)
    def _update_index(self, tensor_id: str, meta_tuple: MetadataTuple) -> bool:
        try:
            coord_list=get_coord_list_from_meta(meta_tuple); coord_obj=TensorCoordinate(*coord_list) if len(coord_list)==6 else None
            if not coord_obj: raise ValueError("Invalid coord list in tuple")
            type_code=get_type_code_from_meta(meta_tuple); status_code=get_status_code_from_meta(meta_tuple)
            tensor_type_str=REVERSE_DATA_TYPE_MAPPING.get(type_code,"unknown"); status_str=REVERSE_STATUS_MAPPING.get(status_code,"unknown")
            meta_file_path=self._get_meta_file_path(coord_obj, tensor_id); relative_meta_path=str(meta_file_path.relative_to(self.db_root_path))
            index_entry = { 'path': relative_meta_path, 'type': tensor_type_str, 'stat': status_str, 'g': coord_list[1], 'l': coord_list[0], 'n': coord_list[2] }
            print(f"DEBUG INDEX UPDATE: Adding/Updating ID {tensor_id} -> Type: {tensor_type_str}, Status: {status_str}, Coords: {coord_obj}")
            self.index[tensor_id] = index_entry
            self._index_dirty = True
            return True
        except Exception as e: print(f"---!!! ERROR updating index for {tensor_id} from tuple: {e} !!!---"); return False
    def _remove_from_index(self, tensor_id: str):
         if tensor_id in self.index: del self.index[tensor_id]; self._index_dirty = True

    # --- Основные Публичные Методы ---
    # (insert_veector_tensor без изменений, т.к. он уже работает с кортежем)
    def insert_veector_tensor(self,
                              meta_tuple: MetadataTuple, # Принимает КОРТЕЖ
                              knowledge_data: Any = None
                             ) -> Optional[str]:
        """
        Сохраняет тензор (КОРТЕЖ метаданных + данные). Генерирует ID по КОРТЕЖУ.
        Сохраняет КОРТЕЖ в .meta. Обновляет индекс по КОРТЕЖУ.
        """
        tensor_id = None
        try:
            # Валидируем кортеж перед использованием
            if not validate_tensor_tuple(meta_tuple):
                 print("Error (insert): Invalid metadata tuple provided."); return None

            # Генерируем ID тензора по КОРТЕЖУ
            tensor_id = get_tensor_hash(meta_tuple) # Ожидает КОРТЕЖ

            # Получение данных из КОРТЕЖА
            coord_obj = get_coord_obj_from_meta(meta_tuple)
            has_blob_flag = get_has_blob_flag_from_meta(meta_tuple)
            tags = get_tags_list_from_meta(meta_tuple)
            dtype_code = get_dtype_code_from_meta(meta_tuple)
            if not coord_obj: raise ValueError("Invalid coordinate object from tuple")

        except Exception as e_extract: print(f"Error (insert): Cannot extract info/hash from tuple: {e_extract}"); return None

        meta_file_path = self._get_meta_file_path(coord_obj, tensor_id)

        # Обработка данных блоба
        blob_saved = False; blob_path = None; blob_format = None
        if has_blob_flag == 1:
            if knowledge_data is None: print(f"Error (insert): Blob data missing for {tensor_id}."); return None

            is_int8 = (dtype_code == DTYPE_MAPPING.get('int8'))
            is_embed_or_lm = (TAG_COMP_EMBEDDING in tags) or (TAG_COMP_LM_HEAD in tags)
            if is_embed_or_lm or is_int8: blob_format = 'npy'
            else: blob_format = 'pickle'

            if blob_format == 'npy':
                if not isinstance(knowledge_data, np.ndarray):
                     try: knowledge_data = np.array(knowledge_data)
                     except Exception as np_conv_e: print(f"Error: Failed NPY conversion: {np_conv_e}"); return None
                if is_int8 and knowledge_data.dtype != np.int8:
                     print(f"Warn: INT8 specified but data is {knowledge_data.dtype}. Casting.")
                     knowledge_data = knowledge_data.astype(np.int8)
                store_func = self._store_npy_blob
            else: store_func = self._store_pickle_blob

            blob_path = self._get_blob_file_path(coord_obj, tensor_id, blob_format)
            blob_saved = store_func(knowledge_data, blob_path)
            if not blob_saved: print(f"---!!! ERROR: Blob store failed for {tensor_id} !!!---"); return None
        elif knowledge_data is not None: print(f"Warn (insert): Data provided for {tensor_id}, but no blob expected.")

        if not self._save_meta_tuple_to_file(meta_tuple, meta_file_path):
            if blob_saved and blob_path and blob_path.is_file():
                try: blob_path.unlink(missing_ok=True)
                except OSError as unlink_e: print(f"Error rolling back blob: {unlink_e}")
            return None

        if not self._update_index(tensor_id, meta_tuple):
             print(f"---!!! CRITICAL ERROR: Index update FAILED for {tensor_id}. Rolling back saves !!!---")
             try: meta_file_path.unlink(missing_ok=True)
             except OSError as unlink_e: print(f"Error rolling back meta: {unlink_e}")
             if blob_saved and blob_path and blob_path.is_file():
                 try: blob_path.unlink(missing_ok=True)
                 except OSError as unlink_e: print(f"Error rolling back blob: {unlink_e}")
             return None
        return tensor_id

    # --- get_veector_tensor: Возвращает СТРУКТУРУ СПИСКА ---
    def get_veector_tensor(self,
                           doc_id: str,
                           load_knowledge_data: bool = False,
                           use_mmap: bool = True
                          ) -> Optional[List]: # ВОЗВРАЩАЕМ СПИСОК
        """
        v0.9.7: Загружает КОРТЕЖ из .meta, РЕКОНСТРУИРУЕТ СТРУКТУРУ СПИСКА,
                включая metadata_extra, и возвращает ее.
        """
        index_entry = self.index.get(doc_id)
        if not index_entry: return None
        meta_file_path = self.db_root_path / index_entry['path']

        # 1. Загружаем КОРТЕЖ метаданных
        meta_tuple = self._load_meta_tuple_from_file(meta_file_path)
        if not meta_tuple:
            print(f"Warn: Meta tuple missing/invalid for {doc_id}. Removing from index.")
            self._remove_from_index(doc_id); return None

        # 2. Реконструируем СТАРУЮ структуру списка
        tensor_structure: List = []
        knowledge_data: Any = None
        try:
            # Извлекаем компоненты из кортежа с помощью геттеров
            data_desc = get_data_description_from_meta(meta_tuple)
            coord_obj = get_coord_obj_from_meta(meta_tuple)
            shape_list = get_shape_list_from_meta(meta_tuple)
            tags_list = get_tags_list_from_meta(meta_tuple)
            ops_seq = get_ops_sequences_from_meta(meta_tuple)
            interface = get_interface_from_meta(meta_tuple)
            filters = get_filters_from_meta(meta_tuple)
            exit_gates = get_exit_gates_from_meta(meta_tuple)
            lifecycle = get_lifecycle_list_from_meta(meta_tuple)
            parents = get_parents_list_from_meta(meta_tuple)
            metadata_extra = get_metadata_extra_from_meta(meta_tuple) # <<< ДОБАВЛЕНО >>> Извлекаем metadata_extra

            # Проверка координатного объекта
            if not coord_obj: raise ValueError("Failed to reconstruct coordinate object")

            has_blob_flag = data_desc[4]; type_code = data_desc[1]; dtype_code = data_desc[2];
            name_id = data_desc[3]; version = data_desc[0]; status_code = lifecycle[0]; evo_version = lifecycle[1]

            # Собираем старый MetaDict
            meta_dict = {
                "evolutionary_version": evo_version, "parents": parents,
                "status": REVERSE_STATUS_MAPPING.get(status_code, "unknown"),
                "tensor_type": REVERSE_DATA_TYPE_MAPPING.get(type_code, "unknown"),
                "created_at": "N/A",
                "coordinate_str": coord_obj.to_string(), "tags": tags_list,
                "interface": interface or {}, "ops_sequences": ops_seq or {},
                "has_blob_data": (has_blob_flag == 1),
                "dtype": REVERSE_DTYPE_MAPPING.get(dtype_code, None),
                "shape": tuple(shape_list) if shape_list else None,
                "data_hash": None,
                "_encoded_metadata_v1_": meta_tuple # Встраиваем сам кортеж
            }
            # <<< ИЗМЕНЕНО >>> Добавляем извлеченный metadata_extra в meta_dict
            if metadata_extra:
                meta_dict.update(metadata_extra)

            # Заглушки для OpChan, Filters, Gates
            op_channels_section = [[9,0,0], [], []] # Identity заглушка
            filters_section = filters or []
            exit_gates_section = exit_gates or []
            # Собираем базовую структуру списка
            tensor_structure = [ coord_obj, op_channels_section, filters_section, exit_gates_section, meta_dict ]

            # 3. Загружаем блоб, если нужно
            if has_blob_flag == 1 and load_knowledge_data:
                is_int8 = (dtype_code == DTYPE_MAPPING.get('int8'))
                is_embed_or_lm = (TAG_COMP_EMBEDDING in tags_list) or (TAG_COMP_LM_HEAD in tags_list)
                blob_format = 'npy' if (is_embed_or_lm or is_int8) else 'pickle'
                blob_file_path = self._get_blob_file_path(coord_obj, doc_id, blob_format)
                load_func = self._load_npy_blob if blob_format == 'npy' else self._load_pickle_blob
                load_args = {'use_mmap': use_mmap} if blob_format == 'npy' else {}
                knowledge_data = load_func(blob_file_path, **load_args)
                if knowledge_data is not None:
                    tensor_structure.append(knowledge_data)
                    # Обновляем meta_dict актуальными данными из блоба (уже сделано выше)
                    meta_dict["dtype"] = str(getattr(knowledge_data, 'dtype', type(knowledge_data).__name__))
                    meta_dict["shape"] = tuple(getattr(knowledge_data, 'shape', None))
                    meta_dict["has_blob_data"] = True
                    # <<< ИЗМЕНЕНО >>> Теперь scale должен быть в meta_dict, если он был в metadata_extra
                    if is_int8 and "quantization_scale" not in meta_dict:
                         print(f"WARN: Loaded INT8 blob for {doc_id} but scale not found in reconstructed meta_dict (should be in metadata_extra).")
                else:
                     print(f"Warn: Failed load blob for {doc_id}. Returning structure without data.")
                     meta_dict["has_blob_data"] = False

            # 4. Валидируем финальную структуру списка
            if not validate_tensor(tensor_structure):
                 print(f"Warn: Reconstructed list structure failed validation for {doc_id}.")
                 return None

        except Exception as e: print(f"Error reconstructing list structure for {doc_id}: {e}"); traceback.print_exc(); return None

        return tensor_structure # Возвращаем структуру списка

    # --- Методы архивации/обновления/поиска (Продолжают работать со списком для совместимости) ---

    # --- Вспомогательные функции для сохранения/загрузки списка (для archive/update) ---
    def _load_list_structure_from_file(self, meta_file_path: Path) -> Optional[List]:
        """ Загружает СТРУКТУРУ СПИСКА из файла (для archive/update). """
        # Эта функция нужна, т.к. archive/update работают со старой структурой
        if not meta_file_path.is_file(): return None
        try:
             with open(meta_file_path, 'rb') as f: structure = pickle.load(f)
             # Важно: Проверяем, это старая структура списка или новый кортеж?
             if validate_tensor(structure): # Если это список
                  return structure
             elif validate_tensor_tuple(structure): # Если это кортеж (на случай, если файл был сохранен новой версией)
                  print(f"Warn (_load_list): Found tuple in {meta_file_path}, attempting reconstruction.")
                  # Вызываем get_veector_tensor, чтобы реконструировать список из кортежа
                  # Нам нужен ID, которого здесь нет. Проще вернуть None и пересохранить.
                  # Или можно попробовать извлечь ID из пути файла, но это ненадежно.
                  # Пока просто возвращаем None, если находим кортеж там, где ожидали список.
                  print(f"Error (_load_list): Cannot reconstruct list from tuple without ID in {meta_file_path}.")
                  return None
             else:
                  print(f"Warn (_load_list): Invalid structure loaded from {meta_file_path}")
                  return None
        except Exception as e: print(f"Error (_load_list) loading structure file {meta_file_path}: {e}"); return None

    def _save_list_structure_to_file(self, list_structure: List, meta_file_path: Path) -> bool:
         """ Сохраняет СТРУКТУРУ СПИСКА в файл (для archive/update). """
         # Эта функция нужна, т.к. archive/update работают со старой структурой
         try:
             if not validate_tensor(list_structure):
                  print(f"Error (_save_list): Invalid list structure provided."); return False
             meta_file_path.parent.mkdir(parents=True, exist_ok=True)
             with open(meta_file_path, 'wb') as f: pickle.dump(list_structure, f, pickle.HIGHEST_PROTOCOL)
             return True
         except Exception as e: print(f"Error (_save_list) writing structure file {meta_file_path}: {e}"); return False

    # --- Обновляем archive_tensor / update_tensor_metadata для работы со списком ---
    def archive_tensor(self, doc_id: str) -> bool:
        tensor_structure = self.get_veector_tensor(doc_id, load_knowledge_data=False); # Загружаем список
        if not tensor_structure: print(f"Error (archive): Cannot load tensor {doc_id}"); return False
        try:
            meta_dict = get_tensor_metadata(tensor_structure); # Получаем meta_dict из списка
            if meta_dict.get("status") == "archived": return True # Уже архивирован

            meta_dict["status"] = "archived"; meta_dict["archived_at"] = datetime.now().isoformat()

            # --- Обновляем КОРТЕЖ внутри meta_dict ---
            meta_tuple = meta_dict.get("_encoded_metadata_v1_")
            updated_meta_tuple = None
            if meta_tuple and validate_tensor_tuple(meta_tuple): # Проверяем наличие и валидность кортежа
                try:
                    lifecycle_list = list(get_lifecycle_list_from_meta(meta_tuple)); lifecycle_list[0] = STATUS_MAPPING["archived"];
                    temp_list = list(meta_tuple); temp_list[8] = lifecycle_list; # Обновляем статус в кортеже
                    updated_meta_tuple = tuple(temp_list)
                    meta_dict["_encoded_metadata_v1_"] = updated_meta_tuple # Сохраняем обновленный кортеж в meta_dict
                except Exception as tuple_update_e: print(f"Warn (archive): Failed to update meta tuple for {doc_id}: {tuple_update_e}")
            else: print(f"Warn (archive): Meta tuple missing or invalid in {doc_id}, cannot update status within tuple.")
            # --- Конец обновления кортежа ---

            coord_obj = get_tensor_coord(tensor_structure);
            if not coord_obj: raise ValueError("Cannot get coords from list structure")
            meta_file_path = self._get_meta_file_path(coord_obj, doc_id)

            # Сохраняем КОРТЕЖ метаданных, если он был обновлен, иначе ничего не делаем с файлом
            saved_meta = False
            if updated_meta_tuple:
                 saved_meta = self._save_meta_tuple_to_file(updated_meta_tuple, meta_file_path)
                 if not saved_meta: print(f"ERROR (archive): Failed save updated meta tuple for {doc_id}"); return False
            else:
                 print(f"Warn (archive): Meta tuple was not updated, meta file for {doc_id} not saved.")
                 # В этом случае индекс тоже не стоит обновлять? Или обновить по meta_dict?
                 # Пока обновляем индекс по meta_dict в любом случае.

            # Обновляем индекс (по КОРТЕЖУ, если он есть, иначе как получится)
            if self._update_index(doc_id, updated_meta_tuple if updated_meta_tuple else meta_tuple): return True
            else: print(f"CRITICAL ERROR (archive): Failed index update for {doc_id}"); return False

        except Exception as e: print(f"Error archiving {doc_id}: {e}"); traceback.print_exc(); return False

    def update_tensor_metadata(self, doc_id: str, updates: Dict) -> bool:
         print("Warning: update_tensor_metadata currently only supports 'status' updates.")
         if 'status' in updates:
             target_status = updates['status']
             if target_status == 'archived': return self.archive_tensor(doc_id)
             elif target_status == 'active': # Активация
                  tensor_structure = self.get_veector_tensor(doc_id, load_knowledge_data=False); # Загружаем список
                  if not tensor_structure: print(f"Error (activate): Cannot load tensor {doc_id}"); return False
                  try:
                      meta_dict = get_tensor_metadata(tensor_structure); # Получаем meta_dict
                      if meta_dict.get("status") == "active": return True # Уже активен

                      meta_dict["status"] = "active"

                      # --- Обновляем КОРТЕЖ внутри meta_dict ---
                      meta_tuple = meta_dict.get("_encoded_metadata_v1_")
                      updated_meta_tuple = None
                      if meta_tuple and validate_tensor_tuple(meta_tuple): # Проверяем кортеж
                          try:
                              lifecycle_list = list(get_lifecycle_list_from_meta(meta_tuple)); lifecycle_list[0] = STATUS_MAPPING["active"];
                              temp_list = list(meta_tuple); temp_list[8] = lifecycle_list; # Обновляем статус в кортеже
                              updated_meta_tuple = tuple(temp_list)
                              meta_dict["_encoded_metadata_v1_"] = updated_meta_tuple # Сохраняем обновленный кортеж
                          except Exception as tuple_update_e: print(f"Warn (activate): Failed to update meta tuple for {doc_id}: {tuple_update_e}")
                      else: print(f"Warn (activate): Meta tuple missing or invalid in {doc_id}, cannot update status within tuple.")
                      # --- Конец обновления кортежа ---

                      coord_obj = get_tensor_coord(tensor_structure);
                      if not coord_obj: raise ValueError("Cannot get coords from list structure")
                      meta_file_path = self._get_meta_file_path(coord_obj, doc_id)

                      # Сохраняем КОРТЕЖ метаданных, если он был обновлен
                      saved_meta = False
                      if updated_meta_tuple:
                           saved_meta = self._save_meta_tuple_to_file(updated_meta_tuple, meta_file_path)
                           if not saved_meta: print(f"ERROR (activate): Failed save updated meta tuple for {doc_id}"); return False
                      else:
                           print(f"Warn (activate): Meta tuple was not updated, meta file for {doc_id} not saved.")

                      # Обновляем индекс (по КОРТЕЖУ)
                      if self._update_index(doc_id, updated_meta_tuple if updated_meta_tuple else meta_tuple):
                          print(f"Tensor {doc_id} activated."); return True
                      else: print(f"CRITICAL ERROR (activate): Failed index update for {doc_id}"); return False

                  except Exception as e: print(f"Error activating {doc_id}: {e}"); traceback.print_exc(); return False
             else: print(f"Error: Unknown status '{target_status}'"); return False
         else: print("Error: Only 'status' updates are supported."); return False


    def find_tensors(self, criteria_func = None) -> Dict[str, List]: # Возвращает Dict[ID, ListStructure]
        # (Без изменений)
        results = {}
        for doc_id in list(self.index.keys()):
             index_entry = self.index.get(doc_id);
             if not index_entry: continue
             list_structure = None
             try: list_structure = self.get_veector_tensor(doc_id, load_knowledge_data=False)
             except Exception as e: print(f"Error loading {doc_id} during find: {e}"); continue
             if list_structure:
                  try:
                      if criteria_func is None or criteria_func(doc_id, list_structure): results[doc_id] = list_structure
                  except Exception as crit_e: print(f"Error applying criteria to {doc_id}: {crit_e}")
        return results

    def find_active_tensors(
        self,
        tensor_type: Optional[str] = None,
        tags: Optional[List[int]] = None,
        coord_filter: Optional[Dict] = None
    ) -> Dict[str, List]:  # Возвращает Dict[ID, ListStructure]
        # (Без изменений)
        results = {}
        query_tags_set = set(tags) if tags else None
        for doc_id, index_entry in self.index.items():
            if index_entry.get('stat') != 'active': continue
            if tensor_type and index_entry.get('type') != tensor_type: continue
            if coord_filter:
                coord_match = True
                if 'group' in coord_filter and index_entry.get('g') != coord_filter['group']: coord_match = False
                if coord_match and 'layer' in coord_filter and index_entry.get('l') != coord_filter['layer']: coord_match = False
                if coord_match and 'nest' in coord_filter and index_entry.get('n') != coord_filter['nest']: coord_match = False
                if not coord_match: continue
            list_structure_for_tags = None
            if query_tags_set:
                list_structure_for_tags = self.get_veector_tensor(doc_id, load_knowledge_data=False)
                if not list_structure_for_tags: continue
                try:
                    tensor_tags_list = get_tensor_tags(list_structure_for_tags)
                    if not query_tags_set.issubset(set(tensor_tags_list)): continue
                except Exception as e: print(f"Error getting tags for {doc_id}: {e}"); continue
            final_list_structure = (list_structure_for_tags if list_structure_for_tags is not None else self.get_veector_tensor(doc_id, load_knowledge_data=False))
            if final_list_structure: results[doc_id] = final_list_structure
        return results

    def find_children(self, parent_doc_id: str) -> List[str]:  # Возвращает List[ID]
        # (Без изменений)
        children_ids = []
        for doc_id in list(self.index.keys()):
            list_structure = self.get_veector_tensor(doc_id, load_knowledge_data=False)
            if list_structure:
                try:
                    parents = get_tensor_parents(list_structure)
                    if parents and parent_doc_id in parents: children_ids.append(doc_id)
                except Exception as e: print(f"Error getting parents for {doc_id}: {e}"); continue
        return children_ids

    def delete_tensor(self, tensor_id: str, delete_blob: bool = True) -> bool:
         """ Удаляет тензор (работает со списком). """
         # (Логика без изменений, но использует get_veector_tensor для загрузки списка)
         index_entry = self.index.get(tensor_id);
         if not index_entry: print(f"Info (delete): Tensor {tensor_id} not in index."); return False
         meta_file_path = self.db_root_path / index_entry['path']

         # Попытаемся загрузить структуру, чтобы определить, есть ли блоб
         list_structure = self.get_veector_tensor(tensor_id, load_knowledge_data=False) # Загружаем список

         # Удаляем блоб, если он есть и нужно
         if delete_blob and list_structure:
              try:
                  meta_dict = get_tensor_metadata(list_structure)
                  if meta_dict.get("has_blob_data"):
                       coord_obj = get_tensor_coord(list_structure); tags = get_tensor_tags(list_structure);
                       # Получаем dtype_code из meta_dict или из кортежа внутри
                       dtype_code = DTYPE_MAPPING.get(meta_dict.get("dtype"), 0)
                       if dtype_code == 0 and "_encoded_metadata_v1_" in meta_dict:
                            meta_tuple = meta_dict["_encoded_metadata_v1_"]
                            if validate_tensor_tuple(meta_tuple):
                                 dtype_code = get_dtype_code_from_meta(meta_tuple)

                       if coord_obj:
                            is_int8 = (dtype_code == DTYPE_MAPPING.get('int8'))
                            is_embed_or_lm = (TAG_COMP_EMBEDDING in tags) or (TAG_COMP_LM_HEAD in tags)
                            blob_format = 'npy' if (is_embed_or_lm or is_int8) else 'pickle'
                            blob_file_path = self._get_blob_file_path(coord_obj, tensor_id, blob_format)
                            if blob_file_path.is_file(): blob_file_path.unlink(); print(f"Deleted blob: {blob_file_path}")
                            else: print(f"Warn (delete): Blob file not found at {blob_file_path}")
                       else: print(f"Warn (delete): Cannot determine blob path for {tensor_id} (no coords).")
              except Exception as e: print(f"Error processing blob for deletion ({tensor_id}): {e}")
         elif delete_blob and not list_structure:
              print(f"Warn (delete): Cannot load structure for {tensor_id} to check for blob.")

         # Удаляем мета-файл
         if meta_file_path.is_file():
              try: meta_file_path.unlink(); print(f"Deleted meta: {meta_file_path}")
              except OSError as e: print(f"Error deleting meta file {meta_file_path}: {e}")
         else: print(f"Warn (delete): Meta file not found at {meta_file_path}")

         # Удаляем из индекса
         self._remove_from_index(tensor_id)
         print(f"Deleted tensor {tensor_id} from index.")
         return True
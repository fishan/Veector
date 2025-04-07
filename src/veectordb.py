# FILE: veectordb.py
# Version: 0.9.8 (Added save_index_as, modified __init__/__load_index for initial path)

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
VEECTORDB_VERSION = "0.9.8" # Added save_index_as, modified __init__/__load_index

# --- Импорты из tensors.py (v0.7.6+ требуется) ---
TENSORS_VERSION_REQ = "0.7.6"
try:
    # Проверяем версию tensors перед импортом остального
    from tensors import TENSORS_VERSION
    if TENSORS_VERSION < TENSORS_VERSION_REQ:
         raise ImportError(f"VeectorDB v{VEECTORDB_VERSION} requires tensors.py v{TENSORS_VERSION_REQ}+, found v{TENSORS_VERSION}")

    from tensors import (
        TensorCoordinate, MetadataTuple, validate_tensor, validate_tensor_tuple,
        get_tensor_hash, get_data_description_from_meta, get_version_from_meta,
        get_type_code_from_meta, get_dtype_code_from_meta, get_name_id_from_meta,
        get_has_blob_flag_from_meta, get_coord_list_from_meta, get_coord_obj_from_meta,
        get_shape_list_from_meta, get_tags_list_from_meta, get_ops_sequences_from_meta,
        get_interface_from_meta, get_filters_from_meta, get_exit_gates_from_meta,
        get_lifecycle_list_from_meta, get_status_code_from_meta, get_evo_version_from_meta,
        get_parents_list_from_meta, get_metadata_extra_from_meta, get_tensor_metadata,
        get_tensor_coord, get_tensor_type, get_tensor_status, get_tensor_tags,
        has_blob_data, get_tensor_parents, get_tensor_op_channels, get_tensor_filters,
        get_tensor_exit_gates, TAG_TYPE_PROCESSOR, TAG_TYPE_KNOWLEDGE, TAG_TYPE_CONVERTER,
        TAG_TYPE_STATE, TAG_MODEL_QWEN2, TAG_MODEL_LLAMA3, TAG_MODEL_DEEPSEEK,
        TAG_PREC_FLOAT32, TAG_PREC_FLOAT16, TAG_PREC_BFLOAT16, TAG_PREC_INT8,
        TAG_PREC_INT4, TAG_COMP_WEIGHTS, TAG_COMP_BIAS, TAG_COMP_EMBEDDING,
        TAG_COMP_ATTN_Q, TAG_COMP_ATTN_K, TAG_COMP_ATTN_V, TAG_COMP_ATTN_O,
        TAG_COMP_ATTN_QKV, TAG_COMP_FFN_GATE, TAG_COMP_FFN_UP, TAG_COMP_FFN_DOWN,
        TAG_COMP_LAYERNORM, TAG_COMP_LM_HEAD, DATA_TYPE_MAPPING, REVERSE_DATA_TYPE_MAPPING,
        DTYPE_MAPPING, REVERSE_DTYPE_MAPPING, STATUS_MAPPING, REVERSE_STATUS_MAPPING
    )
    print(f"  [VeectorDB] Successfully imported tensors v{TENSORS_VERSION}")

except ImportError as e:
    print(f"---!!! FATAL ERROR (VeectorDB): Cannot import required components from tensors.py (v{TENSORS_VERSION_REQ}+): {e} !!!---")
    raise # Прерываем выполнение, если импорт не удался
except Exception as e_other:
    print(f"---!!! FATAL ERROR (VeectorDB): Unexpected error importing from tensors.py: {e_other} !!!---")
    traceback.print_exc()
    raise

class VeectorDB:
    """
    Veector Database v0.9.8
    - Позволяет загружать начальный индекс из указанного файла.
    - Позволяет сохранять индекс в указанный файл (save_index_as).
    - Стандартное сохранение (при close/__del__) происходит в основной index_path.
    """
    INDEX_FILENAME = "tensor_index.pkl" # Имя файла индекса по умолчанию

    def __init__(self, db_dir: Union[str, Path] = "data/db", initial_index_path: Optional[Union[str, Path]] = None):
        """
        Инициализирует VeectorDB.

        Args:
            db_dir: Путь к директории базы данных.
            initial_index_path: Опциональный путь к файлу индекса для начальной загрузки.
                                Если None, загружается стандартный INDEX_FILENAME.
        """
        print(f"--- Initializing VeectorDB v{VEECTORDB_VERSION} ---")
        self.db_root_path = Path(db_dir).resolve()
        # Путь для сохранения по умолчанию (_save_index, close, __del__)
        self.index_path = self.db_root_path / self.INDEX_FILENAME
        self.index: Dict[str, Dict] = {}
        self._index_dirty = False # Флаг "грязного" индекса (нужно ли сохранять)
        self.db_root_path.mkdir(parents=True, exist_ok=True)

        # Определяем, какой файл загружать при инициализации
        load_path = Path(initial_index_path).resolve() if initial_index_path else self.index_path
        self._load_index(load_path=load_path) # Передаем путь в _load_index

        print(f"VeectorDB initialized at {self.db_root_path}. Index entries loaded: {len(self.index)} from '{load_path.name}'.")

    def _load_index(self, load_path: Path):
        """Загружает индекс из указанного файла."""
        if load_path.is_file():
            try:
                print(f"DEBUG INDEX LOAD: Attempting to load from {load_path}...")
                with open(load_path, 'rb') as f:
                    self.index = pickle.load(f)
                if not isinstance(self.index, dict):
                    print(f"WARN: Loaded index from {load_path.name} is not a dict. Resetting index.")
                    self.index = {}
                else:
                     print(f"DEBUG INDEX LOAD: Success. Loaded {len(self.index)} entries from {load_path.name}.")
            except EOFError:
                 print(f"WARN: EOFError loading index from {load_path.name} (possibly empty or corrupted file). Resetting index.")
                 self.index = {}
            except pickle.UnpicklingError as pe:
                 print(f"WARN: UnpicklingError loading index from {load_path.name}: {pe}. Resetting index.")
                 self.index = {}
            except Exception as e:
                print(f"Warn: Load index failed from {load_path.name}: {e}. Resetting index.")
                self.index = {}
        else:
            print(f"DEBUG INDEX LOAD: Index file '{load_path.name}' not found. Starting with empty index.")
            self.index = {}
        # После загрузки (или если файл не найден/ошибка) индекс считается "чистым"
        self._index_dirty = False

    def _save_index(self):
        """Сохраняет текущий индекс в основной файл self.index_path, если он 'грязный'."""
        if self._index_dirty:
            save_target_path = self.index_path # Используем путь по умолчанию
            current_index_size_before_save = len(self.index)
            print(f"DEBUG INDEX SAVE: Attempting save to '{save_target_path.name}'.")
            print(f"DEBUG INDEX SAVE: Size in memory BEFORE save: {current_index_size_before_save}")
            try: # Логгирование последних ключей
                last_keys = list(self.index.keys())[-5:]
                print(f"DEBUG INDEX SAVE: Last 5 keys in memory: {last_keys}")
            except Exception:
                print("DEBUG INDEX SAVE: Could not get last keys.")

            try:
                save_target_path.parent.mkdir(parents=True, exist_ok=True)
                # Используем временный файл для атомарности
                temp_save_path = save_target_path.with_suffix(f"{save_target_path.suffix}.tmp")
                with open(temp_save_path, 'wb') as f:
                    pickle.dump(self.index, f, pickle.HIGHEST_PROTOCOL)
                # Переименовываем временный файл в основной
                os.replace(temp_save_path, save_target_path)

                self._index_dirty = False # Сбрасываем флаг только при успешном сохранении
                print(f"DEBUG INDEX SAVE: pickle.dump and rename completed for {save_target_path}.")

                # --- Верификация ---
                print(f"DEBUG INDEX SAVE: Attempting immediate reload for verification...")
                reloaded_index = None
                reloaded_size = 0
                try:
                     with open(save_target_path, 'rb') as f_verify:
                         reloaded_index = pickle.load(f_verify)
                     if isinstance(reloaded_index, dict):
                          reloaded_size = len(reloaded_index)
                          print(f"DEBUG INDEX SAVE: Immediate reload SUCCESS. Size loaded: {reloaded_size}")
                          if reloaded_size != current_index_size_before_save:
                              print(f"---!!! CRITICAL WARNING: Size mismatch after save! Expected {current_index_size_before_save}, Loaded {reloaded_size} !!!---")
                     else:
                         print(f"---!!! CRITICAL WARNING: Reloaded index is not a dict! Type: {type(reloaded_index)} !!!---")
                except Exception as reload_e:
                    print(f"---!!! CRITICAL WARNING: Failed to reload index immediately after saving: {reload_e} !!!---")
                # --- Конец верификации ---

            except Exception as e:
                print(f"---!!! ERROR: Failed to save index file {save_target_path}: {e} !!!---")
                # Попытка удалить временный файл, если он остался
                try:
                    if temp_save_path.exists():
                        temp_save_path.unlink()
                        print(f"  Deleted temporary index file {temp_save_path.name}.")
                except OSError:
                    pass
        # else:
            # print("DEBUG INDEX SAVE: Index not dirty, skipping save.") # Раскомментировать для отладки

        current_index_size_after_save = len(self.index)
        print(f"DEBUG INDEX SAVE: Size in memory AFTER save: {current_index_size_after_save}")

    def save_index_as(self, filepath: Union[str, Path]):
        """Сохраняет текущий индекс в указанный файл."""
        target_path = Path(filepath).resolve()
        current_index_size = len(self.index)
        print(f"DEBUG INDEX SAVE AS: Attempting save to '{target_path.name}'.")
        print(f"DEBUG INDEX SAVE AS: Size in memory: {current_index_size}")
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            # Используем временный файл
            temp_target_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
            with open(temp_target_path, 'wb') as f:
                pickle.dump(self.index, f, pickle.HIGHEST_PROTOCOL)
            os.replace(temp_target_path, target_path) # Атомарное переименование
            print(f"DEBUG INDEX SAVE AS: Index saved successfully to {target_path}.")
            # Важно: НЕ сбрасываем _index_dirty здесь,
            # так как основное сохранение в self.index_path может быть еще впереди.
        except Exception as e:
            print(f"---!!! ERROR: Failed to save index to {target_path}: {e} !!!---")
            # Попытка удалить временный файл
            try:
                if temp_target_path.exists():
                    temp_target_path.unlink()
                    print(f"  Deleted temporary index file {temp_target_path.name}.")
            except OSError:
                pass

    def close(self):
        """Закрывает соединение с БД, сохраняя основной индекс, если он был изменен."""
        print(f"Closing VeectorDB connection (saving default index to '{self.index_path.name}')...")
        self._save_index() # Сохраняет в self.index_path если _index_dirty
        self.index = {} # Очищаем индекс в памяти после сохранения

    def __del__(self):
        """Деструктор, пытается сохранить основной индекс при сборке мусора, если он 'грязный'."""
        # Проверяем наличие атрибутов перед использованием, чтобы избежать ошибок при завершении работы интерпретатора
        save_method_exists = hasattr(self, '_save_index') and callable(self._save_index)
        is_dirty = getattr(self, '_index_dirty', False)
        index_path_name = getattr(getattr(self, 'index_path', None), 'name', 'N/A')

        if save_method_exists and is_dirty:
             # Используем try-except на случай проблем во время завершения работы
             try:
                 print(f"DEBUG: __del__ calling _save_index (saving default index to '{index_path_name}'). Index dirty: {self._index_dirty}")
                 self._save_index()
             except Exception as del_save_e:
                 print(f"ERROR: Exception during __del__._save_index(): {del_save_e}")
        # elif save_method_exists:
             # print(f"DEBUG: __del__ for VeectorDB - index not dirty, not saving.")

    def _update_index(self, tensor_id: str, meta_tuple: MetadataTuple) -> bool:
        """Обновляет запись в словаре индекса и устанавливает флаг _index_dirty."""
        try:
            # Валидация кортежа перед извлечением данных
            if not validate_tensor_tuple(meta_tuple):
                 print(f"---!!! ERROR (_update_index): Invalid metadata tuple provided for {tensor_id} !!!---")
                 return False

            coord_list = get_coord_list_from_meta(meta_tuple)
            coord_obj = TensorCoordinate(*coord_list) if len(coord_list) == 6 else None
            if not coord_obj:
                raise ValueError("Invalid coord list in tuple")

            type_code = get_type_code_from_meta(meta_tuple)
            status_code = get_status_code_from_meta(meta_tuple)
            tensor_type_str = REVERSE_DATA_TYPE_MAPPING.get(type_code, "unknown")
            status_str = REVERSE_STATUS_MAPPING.get(status_code, "unknown")

            # Проверка кодов
            if tensor_type_str == "unknown": print(f"WARN (_update_index): Unknown type code {type_code} for {tensor_id}")
            if status_str == "unknown": print(f"WARN (_update_index): Unknown status code {status_code} for {tensor_id}")

            meta_file_path = self._get_meta_file_path(coord_obj, tensor_id)
            relative_meta_path = str(meta_file_path.relative_to(self.db_root_path))

            index_entry = {
                'path': relative_meta_path,
                'type': tensor_type_str,
                'stat': status_str,
                'g': coord_list[1], # group
                'l': coord_list[0], # layer
                'n': coord_list[2]  # nest
            }

            print(f"DEBUG INDEX UPDATE: Adding/Updating ID {tensor_id} -> Type: {tensor_type_str}, Status: {status_str}, Coords: {coord_obj}")
            self.index[tensor_id] = index_entry
            self._index_dirty = True # Устанавливаем флаг!
            return True
        except Exception as e:
            print(f"---!!! ERROR updating index for {tensor_id} from tuple: {e} !!!---")
            traceback.print_exc() # Печатаем traceback для детальной отладки
            return False

    def _remove_from_index(self, tensor_id: str):
        """Удаляет запись из индекса и устанавливает флаг _index_dirty."""
        if tensor_id in self.index:
            del self.index[tensor_id]
            self._index_dirty = True
            print(f"DEBUG INDEX REMOVE: Removed ID {tensor_id}")
        else:
            print(f"DEBUG INDEX REMOVE: ID {tensor_id} not found in index.")

    # --- Вспомогательные Методы для Путей (без изменений) ---
    def _get_structured_dir_path(self, coord: TensorCoordinate) -> Path:
        if not isinstance(coord, TensorCoordinate):
            raise TypeError("Invalid TensorCoordinate object")
        # Используем f-строки для читаемости
        return self.db_root_path / f"g{coord.group}" / f"l{coord.layer}" / f"n{coord.nest}"

    def _get_meta_file_path(self, coord: TensorCoordinate, tensor_id: str) -> Path:
        dir_path = self._get_structured_dir_path(coord)
        return dir_path / f"{tensor_id}.meta"

    def _get_blob_file_path(self, coord: TensorCoordinate, tensor_id: str, blob_format: str) -> Path:
        dir_path = self._get_structured_dir_path(coord)
        suffix = ".npy" if blob_format == 'npy' else ".blob"
        return dir_path / f"{tensor_id}{suffix}"

    # --- Методы Хранения/Загрузки Блобов (без изменений) ---
    def _store_pickle_blob(self, data: Any, blob_file_path: Path) -> bool:
        if data is None:
            return False
        try:
            serialized_data = pickle.dumps(data)
            compressed_data = zlib.compress(serialized_data)
            blob_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(blob_file_path, "wb") as f:
                f.write(compressed_data)
            return True
        except Exception as e:
            print(f"---!!! ERROR storing pickle blob {blob_file_path}: {e} !!!---")
            return False

    def _store_npy_blob(self, data: np.ndarray, blob_file_path: Path) -> bool:
        if not isinstance(data, np.ndarray):
            print(f"Error (_store_npy_blob): Expected numpy array, got {type(data)}.")
            return False
        try:
            blob_file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(blob_file_path, data, allow_pickle=False)
            return True
        except Exception as e:
            print(f"---!!! ERROR storing npy blob {blob_file_path}: {e} !!!---")
            return False

    def _load_pickle_blob(self, blob_file_path: Path) -> Any | None:
        if not blob_file_path.is_file():
            return None
        try:
            with open(blob_file_path, "rb") as f:
                compressed_data = f.read()
            if not compressed_data:
                return None
            serialized_data = zlib.decompress(compressed_data)
            del compressed_data # Освобождаем память
            data = pickle.loads(serialized_data)
            return data
        except Exception as e:
            print(f"---!!! ERROR loading pickle blob {blob_file_path}: {e} !!!---")
            return None

    def _load_npy_blob(self, blob_file_path: Path, use_mmap: bool = True) -> Any | None:
        if not blob_file_path.is_file():
            return None
        try:
            mmap_mode = 'r' if use_mmap else None
            data = np.load(blob_file_path, mmap_mode=mmap_mode, allow_pickle=False)
            return data
        except Exception as e:
            print(f"---!!! ERROR loading npy blob {blob_file_path}: {e} !!!---")
            return None

    # --- Обработка Файлов Метаданных (Работа с КОРТЕЖЕМ) (без изменений) ---
    def _load_meta_tuple_from_file(self, meta_file_path: Path) -> Optional[MetadataTuple]:
        if not meta_file_path.is_file():
            return None
        try:
            with open(meta_file_path, 'rb') as f:
                meta_tuple = pickle.load(f)
            # Используем актуальный валидатор для кортежа
            if validate_tensor_tuple(meta_tuple):
                return meta_tuple
            else:
                print(f"---!!! WARNING: Invalid metadata tuple loaded from {meta_file_path} !!!---")
                return None
        except Exception as e:
            print(f"---!!! ERROR loading meta tuple file {meta_file_path}: {e} !!!---")
            return None

    def _save_meta_tuple_to_file(self, meta_tuple: MetadataTuple, meta_file_path: Path) -> bool:
        try:
            if not validate_tensor_tuple(meta_tuple):
                 print(f"---!!! ERROR: Attempting to save invalid metadata tuple to {meta_file_path} !!!---")
                 return False
            meta_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Атомарное сохранение
            temp_meta_path = meta_file_path.with_suffix(f"{meta_file_path.suffix}.tmp")
            with open(temp_meta_path, 'wb') as f:
                pickle.dump(meta_tuple, f, pickle.HIGHEST_PROTOCOL)
            os.replace(temp_meta_path, meta_file_path)
            return True
        except Exception as e:
            print(f"---!!! ERROR writing meta tuple file {meta_file_path}: {e} !!!---")
            # Попытка удалить временный файл
            try:
                if temp_meta_path.exists():
                    temp_meta_path.unlink()
            except OSError:
                pass
            return False

    # --- Основные Публичные Методы ---

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
            # 1. Валидация и генерация ID из кортежа
            if not validate_tensor_tuple(meta_tuple):
                 print("Error (insert): Invalid metadata tuple provided.")
                 return None
            tensor_id = get_tensor_hash(meta_tuple) # Ожидает КОРТЕЖ

            # 2. Извлечение необходимых данных из кортежа
            coord_obj = get_coord_obj_from_meta(meta_tuple)
            has_blob_flag = get_has_blob_flag_from_meta(meta_tuple)
            tags = get_tags_list_from_meta(meta_tuple)
            dtype_code = get_dtype_code_from_meta(meta_tuple)
            if not coord_obj:
                raise ValueError("Invalid coordinate object extracted from tuple")

        except Exception as e_extract:
            print(f"Error (insert): Cannot extract info/hash from tuple: {e_extract}")
            traceback.print_exc()
            return None

        meta_file_path = self._get_meta_file_path(coord_obj, tensor_id)

        # 3. Обработка и сохранение данных блоба (если есть)
        blob_saved = False
        blob_path = None
        blob_format = None
        if has_blob_flag == 1:
            if knowledge_data is None:
                print(f"Error (insert): Blob data missing for {tensor_id}, but has_blob_flag is set.")
                return None

            # Определение формата блоба
            is_int8 = (dtype_code == DTYPE_MAPPING.get('int8'))
            # Проверяем теги компонентов
            is_embed_or_lm = (TAG_COMP_EMBEDDING in tags) or (TAG_COMP_LM_HEAD in tags)

            blob_format = 'npy' if (is_embed_or_lm or is_int8) else 'pickle'

            # Сохранение блоба
            if blob_format == 'npy':
                if not isinstance(knowledge_data, np.ndarray):
                     try:
                         knowledge_data = np.array(knowledge_data) # Попытка конвертации
                     except Exception as np_conv_e:
                         print(f"Error (insert): Failed NPY conversion for {tensor_id}: {np_conv_e}")
                         return None
                # Приведение типа для int8, если необходимо
                if is_int8 and knowledge_data.dtype != np.int8:
                     print(f"Warn (insert): INT8 specified but data is {knowledge_data.dtype}. Casting for {tensor_id}.")
                     try:
                         knowledge_data = knowledge_data.astype(np.int8)
                     except Exception as cast_e:
                          print(f"Error (insert): Failed casting to int8 for {tensor_id}: {cast_e}")
                          return None
                store_func = self._store_npy_blob
            else: # blob_format == 'pickle'
                store_func = self._store_pickle_blob

            blob_path = self._get_blob_file_path(coord_obj, tensor_id, blob_format)
            blob_saved = store_func(knowledge_data, blob_path)
            if not blob_saved:
                print(f"---!!! ERROR: Blob store failed for {tensor_id} !!!---")
                return None # Не сохраняем метаданные, если блоб не сохранился
        elif knowledge_data is not None:
            # Данные переданы, но флаг блоба не установлен
            print(f"Warn (insert): Data provided for {tensor_id}, but has_blob_flag is not set. Data will NOT be saved.")

        # 4. Сохранение файла метаданных (кортежа)
        if not self._save_meta_tuple_to_file(meta_tuple, meta_file_path):
            # Если метаданные не сохранились, откатываем сохранение блоба (если он был)
            if blob_saved and blob_path and blob_path.is_file():
                print(f"Rolling back blob save for {tensor_id} due to meta save failure.")
                try:
                    blob_path.unlink(missing_ok=True)
                except OSError as unlink_e:
                    print(f"Error rolling back blob: {unlink_e}")
            return None

        # 5. Обновление индекса
        if not self._update_index(tensor_id, meta_tuple):
             # Если индекс не обновился - критическая ошибка, откатываем все
             print(f"---!!! CRITICAL ERROR: Index update FAILED for {tensor_id}. Rolling back saves !!!---")
             try:
                 meta_file_path.unlink(missing_ok=True)
             except OSError as unlink_e:
                 print(f"Error rolling back meta: {unlink_e}")
             if blob_saved and blob_path and blob_path.is_file():
                 try:
                     blob_path.unlink(missing_ok=True)
                 except OSError as unlink_e:
                     print(f"Error rolling back blob: {unlink_e}")
             return None

        # Если все успешно
        return tensor_id

    def get_veector_tensor(self,
                           doc_id: str,
                           load_knowledge_data: bool = False,
                           use_mmap: bool = True
                          ) -> Optional[List]: # ВОЗВРАЩАЕМ СПИСОК
        """
        v0.9.8: Загружает КОРТЕЖ из .meta, РЕКОНСТРУИРУЕТ СТРУКТУРУ СПИСКА,
                включая metadata_extra, и возвращает ее.
        """
        index_entry = self.index.get(doc_id)
        if not index_entry:
            # print(f"Debug (get): Tensor ID {doc_id} not found in index.")
            return None
        meta_file_path = self.db_root_path / index_entry['path']

        # 1. Загружаем КОРТЕЖ метаданных
        meta_tuple = self._load_meta_tuple_from_file(meta_file_path)
        if not meta_tuple:
            print(f"Warn (get): Meta tuple missing/invalid for {doc_id} at {meta_file_path}. Removing from index.")
            self._remove_from_index(doc_id)
            return None

        # 2. Реконструируем СТАРУЮ структуру списка из кортежа
        tensor_structure: List = []
        knowledge_data: Any = None
        try:
            # Извлекаем компоненты из кортежа с помощью геттеров tensors.py
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
            metadata_extra = get_metadata_extra_from_meta(meta_tuple) # Извлекаем metadata_extra

            # Проверка базовых извлеченных данных
            if not coord_obj: raise ValueError("Failed to reconstruct coordinate object from tuple")
            if not data_desc or len(data_desc) < 5: raise ValueError("Invalid data_description in tuple")
            if not lifecycle or len(lifecycle) < 2: raise ValueError("Invalid lifecycle in tuple")

            has_blob_flag = data_desc[4]
            type_code = data_desc[1]
            dtype_code = data_desc[2]
            # name_id = data_desc[3] # Не используется в структуре списка напрямую
            # version = data_desc[0] # Не используется в структуре списка напрямую
            status_code = lifecycle[0]
            evo_version = lifecycle[1]

            # Собираем старый MetaDict
            meta_dict = {
                "evolutionary_version": evo_version,
                "parents": parents or [], # Используем None или пустой список
                "status": REVERSE_STATUS_MAPPING.get(status_code, "unknown"),
                "tensor_type": REVERSE_DATA_TYPE_MAPPING.get(type_code, "unknown"),
                "created_at": "N/A", # В старой структуре не хранилось
                "coordinate_str": coord_obj.to_string(),
                "tags": tags_list or [],
                "interface": interface or {},
                "ops_sequences": ops_seq or {},
                "has_blob_data": (has_blob_flag == 1),
                "dtype": REVERSE_DTYPE_MAPPING.get(dtype_code, None),
                "shape": tuple(shape_list) if shape_list else None,
                "data_hash": None, # Не хранится в кортеже, не можем восстановить
                "creator_id": None, # Не хранится в кортеже
                "_encoded_metadata_v1_": meta_tuple # Встраиваем сам кортеж для справки
            }
            # Добавляем извлеченный metadata_extra в meta_dict
            if metadata_extra:
                meta_dict.update(metadata_extra)

            # Заглушки для OpChan, Filters, Gates (как в create_tensor)
            # Используем Identity [9,0,0] как заглушку по умолчанию
            op_channels_section = [[9,0,0], [], []]
            filters_section = filters or []
            exit_gates_section = exit_gates or []

            # Собираем базовую структуру списка
            tensor_structure = [
                coord_obj,           # [0]
                op_channels_section, # [1]
                filters_section,     # [2]
                exit_gates_section,  # [3]
                meta_dict            # [4]
            ]

            # 3. Загружаем блоб, если нужно
            if has_blob_flag == 1 and load_knowledge_data:
                # Определение формата блоба (дублируем логику из insert)
                is_int8 = (dtype_code == DTYPE_MAPPING.get('int8'))
                is_embed_or_lm = (TAG_COMP_EMBEDDING in tags_list) or (TAG_COMP_LM_HEAD in tags_list)
                blob_format = 'npy' if (is_embed_or_lm or is_int8) else 'pickle'

                blob_file_path = self._get_blob_file_path(coord_obj, doc_id, blob_format)
                load_func = self._load_npy_blob if blob_format == 'npy' else self._load_pickle_blob
                load_args = {'use_mmap': use_mmap} if blob_format == 'npy' else {}

                knowledge_data = load_func(blob_file_path, **load_args)

                if knowledge_data is not None:
                    tensor_structure.append(knowledge_data) # Добавляем блоб в список [5]
                    # Обновляем meta_dict актуальными данными из блоба
                    # (dtype и shape могут отличаться от сохраненных в .meta, если блоб был изменен)
                    meta_dict["dtype"] = str(getattr(knowledge_data, 'dtype', type(knowledge_data).__name__))
                    meta_dict["shape"] = tuple(getattr(knowledge_data, 'shape', ()))
                    meta_dict["has_blob_data"] = True # Подтверждаем наличие блоба
                    # Проверка наличия scale для int8
                    if is_int8 and "quantization_scale" not in meta_dict:
                         print(f"WARN (get): Loaded INT8 blob for {doc_id} but 'quantization_scale' not found in reconstructed meta_dict.")
                else:
                     # Блоб должен быть, но не загрузился
                     print(f"Warn (get): Failed to load expected blob for {doc_id} from {blob_file_path}. Returning structure without data.")
                     meta_dict["has_blob_data"] = False # Указываем, что блоба нет

            # 4. Валидируем финальную структуру списка
            if not validate_tensor(tensor_structure):
                 print(f"Warn (get): Reconstructed list structure failed validation for {doc_id}.")
                 # Не возвращаем невалидную структуру
                 return None

        except Exception as e:
            print(f"Error reconstructing list structure for {doc_id}: {e}")
            traceback.print_exc()
            return None

        return tensor_structure # Возвращаем реконструированную структуру списка

    # --- Методы архивации/обновления/поиска (Продолжают работать со списком для совместимости) ---
    # Они используют get_veector_tensor, который теперь возвращает список,
    # поэтому их внутренняя логика работы со списком должна остаться корректной.
    # Важно: archive/update должны сохранять обновленный КОРТЕЖ в .meta файл.

    def archive_tensor(self, doc_id: str) -> bool:
        """Архивирует тензор, обновляя статус в .meta файле (кортеже) и индексе."""
        tensor_structure = self.get_veector_tensor(doc_id, load_knowledge_data=False) # Загружаем список
        if not tensor_structure:
            print(f"Error (archive): Cannot load tensor structure for {doc_id}")
            return False
        try:
            meta_dict = get_tensor_metadata(tensor_structure) # Получаем meta_dict из списка
            if meta_dict.get("status") == "archived":
                print(f"Info (archive): Tensor {doc_id} is already archived.")
                return True # Уже архивирован

            # --- Обновляем КОРТЕЖ внутри meta_dict ---
            meta_tuple = meta_dict.get("_encoded_metadata_v1_")
            updated_meta_tuple = None
            if meta_tuple and validate_tensor_tuple(meta_tuple): # Проверяем наличие и валидность кортежа
                try:
                    # Создаем изменяемую копию кортежа (список)
                    temp_list = list(meta_tuple)
                    # Обновляем статус в списке lifecycle
                    lifecycle_list = list(temp_list[8]) # Копируем список lifecycle
                    lifecycle_list[0] = STATUS_MAPPING["archived"] # Устанавливаем статус "archived"
                    temp_list[8] = lifecycle_list # Заменяем старый lifecycle новым
                    # Преобразуем обратно в кортеж
                    updated_meta_tuple = tuple(temp_list)
                    print(f"Debug (archive): Prepared updated meta tuple for {doc_id}.")
                except Exception as tuple_update_e:
                    print(f"Warn (archive): Failed to prepare updated meta tuple for {doc_id}: {tuple_update_e}")
                    # Не прерываем, попробуем обновить только индекс
            else:
                print(f"Warn (archive): Meta tuple missing or invalid in {doc_id}, cannot update status within tuple.")

            coord_obj = get_tensor_coord(tensor_structure)
            if not coord_obj:
                raise ValueError("Cannot get coordinates from list structure")
            meta_file_path = self._get_meta_file_path(coord_obj, doc_id)

            # Сохраняем обновленный КОРТЕЖ метаданных, если он был успешно создан
            saved_meta = False
            if updated_meta_tuple:
                 saved_meta = self._save_meta_tuple_to_file(updated_meta_tuple, meta_file_path)
                 if not saved_meta:
                     print(f"ERROR (archive): Failed save updated meta tuple for {doc_id}")
                     return False # Не обновляем индекс, если мета не сохранились
                 else:
                      print(f"Debug (archive): Successfully saved updated meta tuple for {doc_id}.")
            else:
                 print(f"Warn (archive): Meta tuple was not updated, meta file for {doc_id} not saved.")
                 # Продолжаем, чтобы обновить хотя бы индекс

            # Обновляем индекс (по обновленному КОРТЕЖУ, если он есть)
            tuple_for_index = updated_meta_tuple if updated_meta_tuple else meta_tuple
            if tuple_for_index: # Убедимся, что у нас есть хоть какой-то кортеж для обновления индекса
                if self._update_index(doc_id, tuple_for_index):
                    print(f"Tensor {doc_id} successfully archived (index updated).")
                    return True
                else:
                    print(f"CRITICAL ERROR (archive): Failed index update for {doc_id} after potentially saving meta.")
                    # Попытка отката сохранения мета? Сложно. Оставляем как есть.
                    return False
            else:
                 print(f"CRITICAL ERROR (archive): Cannot update index for {doc_id} as no valid meta tuple is available.")
                 return False

        except Exception as e:
            print(f"Error archiving {doc_id}: {e}")
            traceback.print_exc()
            return False

    def update_tensor_metadata(self, doc_id: str, updates: Dict) -> bool:
         """Обновляет метаданные тензора. Пока поддерживается только 'status'."""
         print("Warning: update_tensor_metadata currently only supports 'status' updates.")
         if 'status' in updates:
             target_status = updates['status'].lower() # Приводим к нижнему регистру
             if target_status == 'archived':
                 return self.archive_tensor(doc_id)
             elif target_status == 'active':
                 # Логика активации (обратная архивации)
                  tensor_structure = self.get_veector_tensor(doc_id, load_knowledge_data=False) # Загружаем список
                  if not tensor_structure:
                      print(f"Error (activate): Cannot load tensor structure for {doc_id}")
                      return False
                  try:
                      meta_dict = get_tensor_metadata(tensor_structure) # Получаем meta_dict
                      if meta_dict.get("status") == "active":
                          print(f"Info (activate): Tensor {doc_id} is already active.")
                          return True # Уже активен

                      # --- Обновляем КОРТЕЖ внутри meta_dict ---
                      meta_tuple = meta_dict.get("_encoded_metadata_v1_")
                      updated_meta_tuple = None
                      if meta_tuple and validate_tensor_tuple(meta_tuple): # Проверяем кортеж
                          try:
                              # Создаем изменяемую копию кортежа (список)
                              temp_list = list(meta_tuple)
                              # Обновляем статус в списке lifecycle
                              lifecycle_list = list(temp_list[8])
                              lifecycle_list[0] = STATUS_MAPPING["active"] # Устанавливаем статус "active"
                              temp_list[8] = lifecycle_list
                              # Преобразуем обратно в кортеж
                              updated_meta_tuple = tuple(temp_list)
                              print(f"Debug (activate): Prepared updated meta tuple for {doc_id}.")
                          except Exception as tuple_update_e:
                              print(f"Warn (activate): Failed to prepare updated meta tuple for {doc_id}: {tuple_update_e}")
                      else:
                          print(f"Warn (activate): Meta tuple missing or invalid in {doc_id}, cannot update status within tuple.")

                      coord_obj = get_tensor_coord(tensor_structure)
                      if not coord_obj:
                          raise ValueError("Cannot get coordinates from list structure")
                      meta_file_path = self._get_meta_file_path(coord_obj, doc_id)

                      # Сохраняем обновленный КОРТЕЖ метаданных, если он был создан
                      saved_meta = False
                      if updated_meta_tuple:
                           saved_meta = self._save_meta_tuple_to_file(updated_meta_tuple, meta_file_path)
                           if not saved_meta:
                               print(f"ERROR (activate): Failed save updated meta tuple for {doc_id}")
                               return False
                           else:
                                print(f"Debug (activate): Successfully saved updated meta tuple for {doc_id}.")
                      else:
                           print(f"Warn (activate): Meta tuple was not updated, meta file for {doc_id} not saved.")

                      # Обновляем индекс (по обновленному КОРТЕЖУ)
                      tuple_for_index = updated_meta_tuple if updated_meta_tuple else meta_tuple
                      if tuple_for_index:
                          if self._update_index(doc_id, tuple_for_index):
                              print(f"Tensor {doc_id} successfully activated (index updated).")
                              return True
                          else:
                              print(f"CRITICAL ERROR (activate): Failed index update for {doc_id} after potentially saving meta.")
                              return False
                      else:
                           print(f"CRITICAL ERROR (activate): Cannot update index for {doc_id} as no valid meta tuple is available.")
                           return False

                  except Exception as e:
                      print(f"Error activating {doc_id}: {e}")
                      traceback.print_exc()
                      return False
             else:
                 print(f"Error (update): Unknown status '{updates['status']}'")
                 return False
         else:
             print("Error (update): Only 'status' updates are supported.")
             return False

    def find_tensors(self, criteria_func=None) -> Dict[str, List]: # Возвращает Dict[ID, ListStructure]
        """Находит тензоры, соответствующие критерию. Критерий применяется к СТРУКТУРЕ СПИСКА."""
        results = {}
        # Итерируем по копии ключей, чтобы избежать проблем при удалении из индекса внутри цикла
        for doc_id in list(self.index.keys()):
             index_entry = self.index.get(doc_id)
             if not index_entry:
                 continue # Запись уже удалена? Пропускаем.

             list_structure = None
             try:
                 # Загружаем структуру списка
                 list_structure = self.get_veector_tensor(doc_id, load_knowledge_data=False)
                 # get_veector_tensor вернет None, если мета-файл не найден или невалиден,
                 # и удалит запись из индекса.
                 if list_structure is None:
                     continue # Пропускаем, если структура не загрузилась

                 # Применяем критерий к загруженной структуре списка
                 if criteria_func is None or criteria_func(doc_id, list_structure):
                     results[doc_id] = list_structure

             except Exception as find_e:
                 # Ловим ошибки при загрузке или применении критерия
                 print(f"Error processing {doc_id} during find_tensors: {find_e}")
                 continue
        return results

    def find_active_tensors(
        self,
        tensor_type: Optional[str] = None,
        tags: Optional[List[int]] = None,
        coord_filter: Optional[Dict] = None
    ) -> Dict[str, List]:  # Возвращает Dict[ID, ListStructure]
        """Находит активные тензоры по типу, тегам и координатам, используя индекс и загружая СТРУКТУРУ СПИСКА."""
        results = {}
        query_tags_set = set(tags) if tags else None

        # Итерируем по копии ключей индекса
        for doc_id in list(self.index.keys()):
            index_entry = self.index.get(doc_id)
            if not index_entry: continue

            # 1. Фильтрация по индексу (быстрая)
            if index_entry.get('stat') != 'active': continue
            if tensor_type and index_entry.get('type') != tensor_type: continue
            if coord_filter:
                coord_match = True
                # Используем .get() с default=None для безопасности
                if 'group' in coord_filter and index_entry.get('g') != coord_filter['group']: coord_match = False
                if coord_match and 'layer' in coord_filter and index_entry.get('l') != coord_filter['layer']: coord_match = False
                if coord_match and 'nest' in coord_filter and index_entry.get('n') != coord_filter['nest']: coord_match = False
                if not coord_match: continue

            # 2. Если фильтры по индексу пройдены, загружаем структуру списка
            list_structure = self.get_veector_tensor(doc_id, load_knowledge_data=False)
            if list_structure is None:
                # get_veector_tensor уже удалил запись из индекса, если была ошибка
                continue

            # 3. Фильтрация по тегам (если нужно, требует загрузки структуры)
            if query_tags_set:
                try:
                    tensor_tags_list = get_tensor_tags(list_structure) # Получаем теги из списка
                    if not query_tags_set.issubset(set(tensor_tags_list)):
                        continue # Не соответствует тегам, пропускаем
                except Exception as e:
                    print(f"Error getting tags for {doc_id} from list structure: {e}")
                    continue # Ошибка при получении тегов, пропускаем

            # Если все проверки пройдены, добавляем структуру списка в результаты
            results[doc_id] = list_structure

        return results

    def find_children(self, parent_doc_id: str) -> List[str]:  # Возвращает List[ID]
        """Находит дочерние тензоры по ID родителя, используя СТРУКТУРУ СПИСКА."""
        children_ids = []
        # Итерируем по копии ключей
        for doc_id in list(self.index.keys()):
            list_structure = self.get_veector_tensor(doc_id, load_knowledge_data=False)
            if list_structure:
                try:
                    # Получаем список родителей из структуры списка
                    parents = get_tensor_parents(list_structure)
                    if parents and parent_doc_id in parents:
                        children_ids.append(doc_id)
                except Exception as e:
                    print(f"Error getting parents for {doc_id} from list structure: {e}")
                    continue
        return children_ids

    def delete_tensor(self, tensor_id: str, delete_blob: bool = True) -> bool:
         """ Удаляет тензор (мета-файл, блоб, запись в индексе). """
         index_entry = self.index.get(tensor_id)
         if not index_entry:
             print(f"Info (delete): Tensor {tensor_id} not in index.")
             return False # Уже удален или не существовал

         meta_file_path = self.db_root_path / index_entry['path']
         coord_obj = None
         has_blob_data_flag = False
         tags_list = []
         dtype_code = 0

         # Пытаемся загрузить кортеж метаданных для получения информации о блобе
         # Не используем get_veector_tensor, чтобы не реконструировать весь список
         meta_tuple = self._load_meta_tuple_from_file(meta_file_path)
         if meta_tuple:
              try:
                  coord_obj = get_coord_obj_from_meta(meta_tuple)
                  has_blob_data_flag = (get_has_blob_flag_from_meta(meta_tuple) == 1)
                  tags_list = get_tags_list_from_meta(meta_tuple)
                  dtype_code = get_dtype_code_from_meta(meta_tuple)
              except Exception as e:
                  print(f"Warn (delete): Error extracting info from meta tuple for {tensor_id}: {e}")
                  # Продолжаем удаление мета-файла и записи индекса
         else:
             print(f"Warn (delete): Could not load meta tuple for {tensor_id}. Attempting index/meta deletion only.")

         # Удаляем блоб, если он есть и нужно
         if delete_blob and has_blob_data_flag and coord_obj:
              try:
                  # Определение формата блоба (как в insert/get)
                  is_int8 = (dtype_code == DTYPE_MAPPING.get('int8'))
                  is_embed_or_lm = (TAG_COMP_EMBEDDING in tags_list) or (TAG_COMP_LM_HEAD in tags_list)
                  blob_format = 'npy' if (is_embed_or_lm or is_int8) else 'pickle'

                  blob_file_path = self._get_blob_file_path(coord_obj, tensor_id, blob_format)
                  if blob_file_path.is_file():
                      blob_file_path.unlink()
                      print(f"Deleted blob: {blob_file_path}")
                  else:
                      print(f"Warn (delete): Blob file not found at {blob_file_path}")
              except Exception as e:
                  print(f"Error processing blob for deletion ({tensor_id}): {e}")
         elif delete_blob and has_blob_data_flag and not coord_obj:
              print(f"Warn (delete): Cannot determine blob path for {tensor_id} (coord info missing).")

         # Удаляем мета-файл
         if meta_file_path.is_file():
              try:
                  meta_file_path.unlink()
                  print(f"Deleted meta: {meta_file_path}")
              except OSError as e:
                  print(f"Error deleting meta file {meta_file_path}: {e}")
                  # Не прерываем, пытаемся удалить из индекса
         else:
             print(f"Warn (delete): Meta file not found at {meta_file_path}")

         # Удаляем из индекса в памяти
         self._remove_from_index(tensor_id) # Устанавливает _index_dirty = True
         print(f"Removed tensor {tensor_id} from index.")

         # Сразу сохраняем индекс после удаления
         self._save_index()

         return True

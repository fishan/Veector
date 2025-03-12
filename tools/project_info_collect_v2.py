import os
import json
from typing import List, Dict, Optional
import platform
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_FILE = "config.json"

class ConfigManager:
    """Класс для управления настройками конфигурации."""
    
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.default_config = {
            "source_paths": [],
            "output_file": "collected_code.txt",
            "dir_tree_file": "directory_tree.txt",
            "extensions": ['.py', '.js', '.html', '.css'],
            "include_dir_structure": True,
            "order_by": None,
            "exclude_dirs": ['.git', 'node_modules', '__pycache__'],
            "tree_dirs": [],
            "combined_output": False
        }
        self.configs = self.load_configs()
        self.current_config_name = None
        self.current_config = self.default_config.copy()

    def load_configs(self) -> Dict:
        """Загружает все конфигурации из файла, если он существует."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Ошибка при загрузке конфигураций из {self.config_file}: {str(e)}")
            print(f"Ошибка при загрузке конфигураций: {str(e)}. Используется пустой набор конфигураций.")
            return {}

    def save_configs(self) -> None:
        """Сохраняет все конфигурации в файл."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.configs, f, indent=4, ensure_ascii=False)
            logging.info(f"Конфигурации сохранены в {self.config_file}")
            print(f"Конфигурации сохранены в {self.config_file}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении конфигураций в {self.config_file}: {str(e)}")
            print(f"Ошибка при сохранении конфигураций: {str(e)}")

    def save_current_config(self, config_name: str, overwrite: bool = False) -> bool:
        """Сохраняет текущую конфигурацию под указанным именем."""
        if config_name in self.configs and not overwrite:
            print(f"Конфигурация с именем '{config_name}' уже существует. Используйте перезапись (overwrite=True) для обновления.")
            return False
        self.configs[config_name] = self.current_config.copy()
        self.save_configs()
        self.current_config_name = config_name
        print(f"Конфигурация сохранена как '{config_name}'.")
        return True

    def load_config_by_name(self, config_name: str) -> bool:
        """Загружает конфигурацию по имени."""
        if config_name in self.configs:
            self.current_config = self.configs[config_name].copy()
            self.current_config_name = config_name
            print(f"Конфигурация '{config_name}' загружена.")
            return True
        print(f"Конфигурация с именем '{config_name}' не найдена.")
        return False

    def list_configs(self) -> List[str]:
        """Возвращает список имен всех сохраненных конфигураций."""
        return list(self.configs.keys())

    def delete_config(self, config_name: str) -> bool:
        """Удаляет конфигурацию по имени."""
        if config_name in self.configs:
            del self.configs[config_name]
            self.save_configs()
            if self.current_config_name == config_name:
                self.current_config_name = None
                self.current_config = self.default_config.copy()
            print(f"Конфигурация '{config_name}' удалена.")
            return True
        print(f"Конфигурация с именем '{config_name}' не найдена.")
        return False

    def update_config(self, key: str, value: any) -> None:
        """Обновляет отдельный параметр текущей конфигурации."""
        self.current_config[key] = value
        print(f"Параметр '{key}' обновлен: {value}")

    def get_current_config(self) -> Dict:
        """Возвращает текущую конфигурацию."""
        return self.current_config

    def get_current_config_name(self) -> Optional[str]:
        """Возвращает имя текущей конфигурации."""
        return self.current_config_name


def clear_screen():
    """Очищает экран консоли."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def print_directory_tree(
    directories: List[str], 
    indent: str = "", 
    exclude_dirs: List[str] = None, 
    output_file: str = None
) -> None:
    """
    Выводит структуру выбранных директорий в виде дерева, исключая указанные папки.
    Может сохранять результат в файл, если указан output_file.
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    def write_line(line: str, file_handle=None) -> None:
        print(line)
        if file_handle:
            file_handle.write(line + "\n")

    file_handle = None
    if output_file:
        try:
            file_handle = open(output_file, 'a', encoding='utf-8')
        except Exception as e:
            logging.error(f"Ошибка при открытии файла {output_file} для записи: {str(e)}")
            print(f"Ошибка при открытии файла {output_file}: {str(e)}")
            return

    try:
        for directory in directories:
            dir_name = os.path.basename(directory)
            if dir_name in exclude_dirs:
                continue
            write_line(f"{indent}+ {dir_name}/", file_handle)
            indent += "  "
            for item in sorted(os.listdir(directory)):
                path = os.path.join(directory, item)
                if os.path.isdir(path):
                    if item not in exclude_dirs:
                        print_directory_tree([path], indent, exclude_dirs, output_file)
                else:
                    write_line(f"{indent}- {item}", file_handle)
    except Exception as e:
        logging.error(f"Ошибка при выводе структуры директорий: {str(e)}")
        print(f"Ошибка при выводе структуры директорий: {str(e)}")
    finally:
        if file_handle:
            file_handle.close()


def read_paths_from_file(file_path: str) -> List[str]:
    """Читает список путей из файла. Каждый путь — на новой строке."""
    paths = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if path:
                    paths.append(path)
    except Exception as e:
        logging.error(f"Ошибка при чтении файла путей {file_path}: {str(e)}")
        print(f"Ошибка при чтении файла путей {file_path}: {str(e)}")
    return paths


def collect_files_content(
    source_paths: List[str],
    output_file: str,
    extensions: List[str] = None,
    include_dir_structure: bool = True,
    order_by: str = None,
    exclude_dirs: List[str] = None,
    tree_dirs: List[str] = None,
    combined_output: bool = False
) -> None:
    """
    Собирает содержимое файлов из указанных путей в один текстовый файл.
    """
    if extensions is None:
        extensions = ['.py', '.js', '.html', '.css']
    if exclude_dirs is None:
        exclude_dirs = []
    if tree_dirs is None:
        tree_dirs = [p for p in source_paths if os.path.isdir(p)]

    # Собираем все файлы, соответствующие критериям
    all_files = []
    for path in source_paths:
        if os.path.isfile(path):
            if any(path.lower().endswith(ext.lower()) for ext in extensions):
                all_files.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for filename in files:
                    if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                        file_path = os.path.join(root, filename)
                        all_files.append(file_path)

    # Сортировка файлов, если указан порядок
    if order_by == 'name':
        all_files.sort(key=lambda x: os.path.basename(x).lower())
    elif order_by == 'mtime':
        all_files.sort(key=lambda x: os.path.getmtime(x))

    # Вывод дерева и содержимого
    tree_output_file = output_file if combined_output else "directory_tree.txt"
    if combined_output and os.path.exists(output_file):
        os.remove(output_file)
    if not combined_output and tree_dirs:
        open(tree_output_file, 'w').close()
        print("\nСтруктура директорий перед сборкой файлов:")
        print_directory_tree(tree_dirs, exclude_dirs=exclude_dirs, output_file=tree_output_file)

    try:
        with open(output_file, 'a' if combined_output else 'w', encoding='utf-8') as outfile:
            if combined_output and tree_dirs:
                print("\nСтруктура директорий перед сборкой файлов:")
                print_directory_tree(tree_dirs, exclude_dirs=exclude_dirs, output_file=output_file)
                outfile.write("\n\n")
            
            for file_path in all_files:
                display_path = file_path if include_dir_structure else os.path.basename(file_path)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        file_content = infile.read()
                        outfile.write(f"===== Код из файла: {display_path} =====\n\n")
                        outfile.write(file_content)
                        outfile.write("\n\n")
                except Exception as e:
                    outfile.write(f"Ошибка при чтении файла {display_path}: {str(e)}\n\n")
    except Exception as e:
        logging.error(f"Ошибка при записи в файл {output_file}: {str(e)}")
        print(f"Ошибка при записи в файл {output_file}: {str(e)}")
    
    print(f"Все файлы собраны в {output_file}")
    if not combined_output and tree_dirs:
        print(f"Структура директорий сохранена в {tree_output_file}")


def browse_paths(current_dir: str, selected_paths: List[str]) -> Optional[str]:
    """Интерактивный браузер файловой системы для выбора пути."""
    while True:
        clear_screen()
        print(f"\nТекущая директория: {current_dir}")
        items = sorted(os.listdir(current_dir))
        files = [item for item in items if os.path.isfile(os.path.join(current_dir, item))]
        dirs = [item for item in items if os.path.isdir(os.path.join(current_dir, item))]
        
        print("Доступные файлы (выбор через номера, множественный выбор через пробел, например '2 4 5'):")
        for i, item in enumerate(files, 1):
            path = os.path.join(current_dir, item)
            selected_mark = "[SELECTED]" if path in selected_paths else ""
            print(f"{i}. {item} (FILE) {selected_mark}")

        print("\nДоступные папки (выбор через буквы, множественный выбор через пробел, например 'a c e'):")
        for i, item in enumerate(dirs, 0):
            path = os.path.join(current_dir, item)
            selected_mark = "[SELECTED]" if path in selected_paths else ""
            letter = chr(97 + i)
            print(f"{letter}. {item} (DIR) {selected_mark}")

        print("\nДействия:")
        print("0. Назад (или выход в главное меню, если в корне)")
        print("1. Выбрать текущую папку")
        print("2. Добавить файлы в выбранные")
        print("3. Убрать файлы из выбранных")
        print("4. Выбрать все файлы в папке")
        print("5. Убрать выбор со всех файлов в папке")
        print("6. Добавить папки в выбранные")
        print("7. Убрать папки из выбранных")
        print("8. Вернуться в главное меню")

        choice = input("Введите номер, букву или действие: ").strip().lower()
        if choice == '0':
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                return None
            current_dir = parent_dir
        elif choice == '1':
            if current_dir not in selected_paths:
                selected_paths.append(current_dir)
                print(f"Папка '{current_dir}' добавлена в выбранные.")
            else:
                print("Папка уже выбрана.")
            input("Нажмите Enter для продолжения...")
        elif choice == '2':
            print("Выберите файлы для добавления (введите номера через пробел, например '2 4 5'):")
            file_choices = input("Введите номера файлов: ").strip().split()
            try:
                for file_choice in file_choices:
                    file_choice = int(file_choice) - 1
                    if 0 <= file_choice < len(files):
                        selected_item = files[file_choice]
                        selected_path = os.path.join(current_dir, selected_item)
                        if selected_path not in selected_paths:
                            selected_paths.append(selected_path)
                            print(f"Файл '{selected_path}' добавлен в выбранные.")
                        else:
                            print(f"Файл '{selected_path}' уже выбран.")
                    else:
                        print(f"Неверный номер '{file_choice + 1}', пропущен.")
            except ValueError:
                print("Введите корректные номера.")
            input("Нажмите Enter для продолжения...")
        elif choice == '3':
            print("Выберите файлы для удаления из выбранных (введите номера через пробел, например '2 4 5'):")
            file_choices = input("Введите номера файлов: ").strip().split()
            try:
                for file_choice in file_choices:
                    file_choice = int(file_choice) - 1
                    if 0 <= file_choice < len(files):
                        selected_item = files[file_choice]
                        selected_path = os.path.join(current_dir, selected_item)
                        if selected_path in selected_paths:
                            selected_paths.remove(selected_path)
                            print(f"Файл '{selected_path}' удален из выбранных.")
                        else:
                            print(f"Файл '{selected_path}' не выбран.")
                    else:
                        print(f"Неверный номер '{file_choice + 1}', пропущен.")
            except ValueError:
                print("Введите корректные номера.")
            input("Нажмите Enter для продолжения...")
        elif choice == '4':
            for item in files:
                path = os.path.join(current_dir, item)
                if path not in selected_paths:
                    selected_paths.append(path)
            print("Все файлы в папке добавлены в выбранные.")
            input("Нажмите Enter для продолжения...")
        elif choice == '5':
            for item in files:
                path = os.path.join(current_dir, item)
                if path in selected_paths:
                    selected_paths.remove(path)
            print("Выбор снят со всех файлов в папке.")
            input("Нажмите Enter для продолжения...")
        elif choice == '6':
            print("Выберите папки для добавления (введите буквы через пробел, например 'a c e'):")
            dir_choices = input("Введите буквы папок: ").strip().split()
            try:
                for dir_choice in dir_choices:
                    dir_index = ord(dir_choice) - 97
                    if 0 <= dir_index < len(dirs):
                        selected_item = dirs[dir_index]
                        selected_path = os.path.join(current_dir, selected_item)
                        if selected_path not in selected_paths:
                            selected_paths.append(selected_path)
                            print(f"Папка '{selected_path}' добавлена в выбранные.")
                        else:
                            print(f"Папка '{selected_path}' уже выбрана.")
                    else:
                        print(f"Неверная буква '{dir_choice}', пропущена.")
            except ValueError:
                print("Введите корректные буквы.")
            input("Нажмите Enter для продолжения...")
        elif choice == '7':
            print("Выберите папки для удаления из выбранных (введите буквы через пробел, например 'a c e'):")
            dir_choices = input("Введите буквы папок: ").strip().split()
            try:
                for dir_choice in dir_choices:
                    dir_index = ord(dir_choice) - 97
                    if 0 <= dir_index < len(dirs):
                        selected_item = dirs[dir_index]
                        selected_path = os.path.join(current_dir, selected_item)
                        if selected_path in selected_paths:
                            selected_paths.remove(selected_path)
                            print(f"Папка '{selected_path}' удалена из выбранных.")
                        else:
                            print(f"Папка '{selected_path}' не выбрана.")
                    else:
                        print(f"Неверная буква '{dir_choice}', пропущена.")
            except ValueError:
                print("Введите корректные буквы.")
            input("Нажмите Enter для продолжения...")
        elif choice == '8':
            return None
        else:
            try:
                choice = int(choice) - 1
                if 0 <= choice < len(files):
                    selected_item = files[choice]
                    selected_path = os.path.join(current_dir, selected_item)
                    if selected_path not in selected_paths:
                        selected_paths.append(selected_path)
                        print(f"Файл '{selected_path}' добавлен в выбранные.")
                    else:
                        print("Файл уже выбран.")
                    input("Нажмите Enter для продолжения...")
                else:
                    print("Неверный номер файла, попробуйте снова.")
                    input("Нажмите Enter для продолжения...")
            except ValueError:
                try:
                    dir_index = ord(choice) - 97
                    if 0 <= dir_index < len(dirs):
                        selected_item = dirs[dir_index]
                        selected_path = os.path.join(current_dir, selected_item)
                        current_dir = selected_path
                    else:
                        print("Неверная буква папки, попробуйте снова.")
                        input("Нажмите Enter для продолжения...")
                except ValueError:
                    print("Введите корректный номер, букву или действие.")
                    input("Нажмите Enter для продолжения...")


def sort_paths_manually(paths: List[str]) -> List[str]:
    """Позволяет вручную сортировать список путей."""
    while True:
        clear_screen()
        print("\nТекущий порядок путей:")
        for i, path in enumerate(paths, 1):
            print(f"{i}. {path}")
        print("\nДействия:")
        print("0. Завершить сортировку")
        print("Введите два номера (через пробел), чтобы поменять элементы местами")

        choice = input("Введите действие или номера: ").strip()
        if choice == '0':
            return paths
        try:
            idx1, idx2 = map(int, choice.split())
            idx1 -= 1
            idx2 -= 1
            if 0 <= idx1 < len(paths) and 0 <= idx2 < len(paths):
                paths[idx1], paths[idx2] = paths[idx2], paths[idx1]
                print(f"Элементы {idx1 + 1} и {idx2 + 1} поменяны местами.")
            else:
                print("Неверные номера, попробуйте снова.")
        except ValueError:
            print("Введите два номера через пробел или '0' для завершения.")
        input("Нажмите Enter для продолжения...")


def interactive_menu(config_manager: ConfigManager) -> Optional[Dict]:
    """Интерактивное меню для настройки параметров."""
    while True:
        clear_screen()
        config = config_manager.get_current_config()
        current_config_name = config_manager.get_current_config_name()
        print(f"\nТекущая конфигурация: {current_config_name if current_config_name else 'Новая (не сохранена)'}")
        print("\nТекущие настройки:")
        print(f"1. Пути к файлам/папкам: {config['source_paths']}")
        print(f"2. Выходной файл: {config['output_file']}")
        print(f"3. Файл для структуры директорий: {config['dir_tree_file']}")
        print(f"4. Расширения файлов: {config['extensions']}")
        print(f"5. Включать структуру папок в выходной файл: {config['include_dir_structure']}")
        print(f"6. Порядок сортировки файлов: {config['order_by']}")
        print(f"7. Исключенные папки: {config['exclude_dirs']}")
        print(f"8. Папки для дерева: {config['tree_dirs']}")
        print(f"9. Комбинированный вывод (дерево и содержимое в одном файле): {config['combined_output']}")
        print("\nВыберите действие:")
        print("0. Начать с нуля (очистить текущую конфигурацию)")
        print("1. Изменить пути к файлам/папкам")
        print("2. Изменить выходной файл")
        print("3. Изменить файл для структуры директорий")
        print("4. Изменить расширения файлов")
        print("5. Переключить включение структуры папок")
        print("6. Изменить порядок сортировки")
        print("7. Изменить исключенные папки")
        print("8. Изменить папки для дерева")
        print("9. Переключить комбинированный вывод")
        print("10. Сохранить текущую конфигурацию")
        print("11. Загрузить существующую конфигурацию")
        print("12. Удалить существующую конфигурацию")
        print("13. Запустить сбор файлов с текущей конфигурацией")
        print("14. Выход без сохранения")
        print("15. Завершить программу")

        choice = input("Введите номер действия: ").strip()

        if choice == '0':
            config_manager.current_config = config_manager.default_config.copy()
            config_manager.current_config_name = None
            print("Текущая конфигурация очищена.")
            input("Нажмите Enter для продолжения...")
        elif choice == '1':
            print("\nТекущие пути:", config['source_paths'])
            print("Выберите действие:")
            print("1. Добавить путь (через браузер файловой системы)")
            print("2. Удалить путь")
            print("3. Загрузить пути из файла")
            print("4. Сортировать пути вручную")
            path_choice = input("Введите номер действия: ").strip()
            if path_choice == '1':
                print("Запускается браузер файловой системы...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                selected_path = browse_paths(current_dir, config['source_paths'])
                if selected_path is None:
                    print("Выбор пути завершен, возвращаемся в главное меню.")
            elif path_choice == '2':
                if config['source_paths']:
                    clear_screen()
                    print("Выберите путь для удаления (введите номер):")
                    for i, path in enumerate(config['source_paths'], 1):
                        print(f"{i}. {path}")
                    path_index = input("Введите номер пути: ").strip()
                    try:
                        path_index = int(path_index) - 1
                        if 0 <= path_index < len(config['source_paths']):
                            removed_path = config['source_paths'].pop(path_index)
                            print(f"Путь '{removed_path}' удален.")
                        else:
                            print("Неверный номер пути.")
                    except ValueError:
                        print("Введите корректный номер.")
                    input("Нажмите Enter для продолжения...")
                else:
                    print("Список путей пуст.")
                    input("Нажмите Enter для продолжения...")
            elif path_choice == '3':
                paths_file = input("Введите путь к файлу с путями: ").strip()
                if os.path.exists(paths_file):
                    config_manager.update_config('source_paths', read_paths_from_file(paths_file))
                    print(f"Пути загружены из файла '{paths_file}'.")
                else:
                    print(f"Файл '{paths_file}' не существует.")
                input("Нажмите Enter для продолжения...")
            elif path_choice == '4':
                if config['source_paths']:
                    config_manager.update_config('source_paths', sort_paths_manually(config['source_paths']))
                else:
                    print("Список путей пуст, сортировка невозможна.")
                input("Нажмите Enter для продолжения...")
        elif choice == '2':
            new_output_file = input("Введите имя выходного файла (по умолчанию 'collected_code.txt'): ").strip()
            config_manager.update_config('output_file', new_output_file if new_output_file else "collected_code.txt")
            print(f"Выходной файл установлен: {config['output_file']}")
            input("Нажмите Enter для продолжения...")
        elif choice == '3':
            new_dir_tree_file = input("Введите имя файла для структуры директорий (или 'None' для отключения, по умолчанию 'directory_tree.txt'): ").strip()
            if new_dir_tree_file.lower() == 'none':
                config_manager.update_config('dir_tree_file', None)
                print("Сохранение структуры директорий отключено.")
            else:
                config_manager.update_config('dir_tree_file', new_dir_tree_file if new_dir_tree_file else "directory_tree.txt")
                print(f"Файл для структуры директорий установлен: {config['dir_tree_file']}")
            input("Нажмите Enter для продолжения...")
        elif choice == '4':
            print("\nТекущие расширения файлов:", config['extensions'])
            new_extensions = input("Введите новые расширения через пробел (например, '.py .js .html .css'): ").strip().split()
            if new_extensions:
                config_manager.update_config('extensions', new_extensions)
                print(f"Расширения файлов обновлены: {config['extensions']}")
            else:
                print("Расширения не изменены.")
            input("Нажмите Enter для продолжения...")
        elif choice == '5':
            config_manager.update_config('include_dir_structure', not config['include_dir_structure'])
            print(f"Включение структуры папок: {config['include_dir_structure']}")
            input("Нажмите Enter для продолжения...")
        elif choice == '6':
            print("\nВыберите порядок сортировки:")
            print("1. Без сортировки (None)")
            print("2. По имени файла (name)")
            print("3. По времени изменения (mtime)")
            print("4. Ручная сортировка")
            sort_choice = input("Введите номер: ").strip()
            if sort_choice == '1':
                config_manager.update_config('order_by', None)
            elif sort_choice == '2':
                config_manager.update_config('order_by', 'name')
            elif sort_choice == '3':
                config_manager.update_config('order_by', 'mtime')
            elif sort_choice == '4':
                if config['source_paths']:
                    config_manager.update_config('source_paths', sort_paths_manually(config['source_paths']))
                else:
                    print("Список путей пуст, сортировка невозможна.")
            else:
                print("Неверный выбор, сортировка не изменена.")
            print(f"Порядок сортировки установлен: {config['order_by']}")
            input("Нажмите Enter для продолжения...")
        elif choice == '7':
            print("\nТекущие исключенные папки:", config['exclude_dirs'])
            print("Выберите действие:")
            print("1. Добавить исключенную папку")
            print("2. Удалить исключенную папку")
            excl_choice = input("Введите номер действия: ").strip()
            if excl_choice == '1':
                new_excl = input("Введите имя папки для исключения (например, '.git'): ").strip()
                if new_excl and new_excl not in config['exclude_dirs']:
                    config['exclude_dirs'].append(new_excl)
                    config_manager.update_config('exclude_dirs', config['exclude_dirs'])
                    print(f"Папка '{new_excl}' добавлена в исключения.")
                else:
                    print("Имя папки пустое или уже существует.")
            elif excl_choice == '2':
                if config['exclude_dirs']:
                    clear_screen()
                    print("Выберите папку для удаления из исключений (введите номер):")
                    for i, excl in enumerate(config['exclude_dirs'], 1):
                        print(f"{i}. {excl}")
                    excl_index = input("Введите номер папки: ").strip()
                    try:
                        excl_index = int(excl_index) - 1
                        if 0 <= excl_index < len(config['exclude_dirs']):
                            removed_excl = config['exclude_dirs'].pop(excl_index)
                            config_manager.update_config('exclude_dirs', config['exclude_dirs'])
                            print(f"Папка '{removed_excl}' удалена из исключений.")
                        else:
                            print("Неверный номер папки.")
                    except ValueError:
                        print("Введите корректный номер.")
                else:
                    print("Список исключенных папок пуст.")
            input("Нажмите Enter для продолжения...")
        elif choice == '8':
            print("\nТекущие папки для дерева:", config['tree_dirs'])
            print("Выберите действие:")
            print("1. Добавить папку для дерева (через браузер файловой системы)")
            print("2. Удалить папку из дерева")
            print("3. Использовать все папки из путей")
            tree_choice = input("Введите номер действия: ").strip()
            if tree_choice == '1':
                print("Запускается браузер файловой системы...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                selected_path = browse_paths(current_dir, config['tree_dirs'])
                if selected_path is None:
                    print("Выбор папки завершен, возвращаемся в главное меню.")
            elif tree_choice == '2':
                if config['tree_dirs']:
                    clear_screen()
                    print("Выберите папку для удаления из дерева (введите номер):")
                    for i, path in enumerate(config['tree_dirs'], 1):
                        print(f"{i}. {path}")
                    tree_index = input("Введите номер папки: ").strip()
                    try:
                        tree_index = int(tree_index) - 1
                        if 0 <= tree_index < len(config['tree_dirs']):
                            removed_path = config['tree_dirs'].pop(tree_index)
                            config_manager.update_config('tree_dirs', config['tree_dirs'])
                            print(f"Папка '{removed_path}' удалена из дерева.")
                        else:
                            print("Неверный номер папки.")
                    except ValueError:
                        print("Введите корректный номер.")
                else:
                    print("Список папок для дерева пуст.")
                input("Нажмите Enter для продолжения...")
            elif tree_choice == '3':
                config_manager.update_config('tree_dirs', [p for p in config['source_paths'] if os.path.isdir(p)])
                print("Все папки из путей добавлены в дерево.")
                input("Нажмите Enter для продолжения...")
        elif choice == '9':
            config_manager.update_config('combined_output', not config['combined_output'])
            print(f"Комбинированный вывод: {config['combined_output']}")
            input("Нажмите Enter для продолжения...")
        elif choice == '10':
            config_name = input("Введите имя для сохранения конфигурации: ").strip()
            if not config_name:
                print("Имя конфигурации не может быть пустым.")
            else:
                if config_name in config_manager.list_configs():
                    overwrite = input(f"Конфигурация '{config_name}' уже существует. Перезаписать? (y/n): ").strip().lower()
                    if overwrite == 'y':
                        config_manager.save_current_config(config_name, overwrite=True)
                    else:
                        print("Сохранение отменено.")
                else:
                    config_manager.save_current_config(config_name)
            input("Нажмите Enter для продолжения...")
        elif choice == '11':
            config_names = config_manager.list_configs()
            if not config_names:
                print("Нет сохраненных конфигураций.")
            else:
                clear_screen()
                print("Доступные конфигурации:")
                for i, name in enumerate(config_names, 1):
                    print(f"{i}. {name}")
                config_index = input("Введите номер конфигурации для загрузки: ").strip()
                try:
                    config_index = int(config_index) - 1
                    if 0 <= config_index < len(config_names):
                        config_manager.load_config_by_name(config_names[config_index])
                    else:
                        print("Неверный номер конфигурации.")
                except ValueError:
                    print("Введите корректный номер.")
            input("Нажмите Enter для продолжения...")
        elif choice == '12':
            config_names = config_manager.list_configs()
            if not config_names:
                print("Нет сохраненных конфигураций.")
            else:
                clear_screen()
                print("Доступные конфигурации:")
                for i, name in enumerate(config_names, 1):
                    print(f"{i}. {name}")
                config_index = input("Введите номер конфигурации для удаления: ").strip()
                try:
                    config_index = int(config_index) - 1
                    if 0 <= config_index < len(config_names):
                        confirm = input(f"Вы уверены, что хотите удалить конфигурацию '{config_names[config_index]}'? (y/n): ").strip().lower()
                        if confirm == 'y':
                            config_manager.delete_config(config_names[config_index])
                        else:
                            print("Удаление отменено.")
                    else:
                        print("Неверный номер конфигурации.")
                except ValueError:
                    print("Введите корректный номер.")
            input("Нажмите Enter для продолжения...")
        elif choice == '13':
            if not config['source_paths']:
                print("Список путей пуст, сбор файлов невозможен. Добавьте пути в настройках.")
            else:
                collect_files_content(
                    source_paths=config['source_paths'],
                    output_file=config['output_file'],
                    extensions=config['extensions'],
                    include_dir_structure=config['include_dir_structure'],
                    order_by=config['order_by'],
                    exclude_dirs=config['exclude_dirs'],
                    tree_dirs=config['tree_dirs'],
                    combined_output=config['combined_output']
                )
            input("Нажмите Enter для продолжения...")
        elif choice == '14':
            print("Выход без сохранения изменений.")
            return None
        elif choice == '15':
            print("Программа завершена.")
            exit(0)
        else:
            print("Неверный выбор, попробуйте снова.")
            input("Нажмите Enter для продолжения...")


def main():
    """Основная функция для настройки и запуска скрипта."""
    config_manager = ConfigManager()

    while True:
        config = interactive_menu(config_manager)
        if config is None:
            break


if __name__ == "__main__":
    main()
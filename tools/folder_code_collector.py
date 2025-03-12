import os

def collect_python_files_to_txt(directory, output_file):
    """
    Собирает содержимое всех .py файлов в указанной папке и записывает их в один текстовый файл.
    Каждый файл отделяется разделителем с указанием имени исходного файла.
    """
    # Открываем выходной файл для записи
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Проходим по всем файлам в директории
        for filename in os.listdir(directory):
            # Проверяем, что файл имеет расширение .py
            if filename.endswith('.py'):
                file_path = os.path.join(directory, filename)
                # Читаем содержимое файла
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        file_content = infile.read()
                        # Записываем разделитель с именем файла
                        outfile.write(f"===== Код из файла: {filename} =====\n\n")
                        # Записываем содержимое файла
                        outfile.write(file_content)
                        # Добавляем пустую строку после кода для читаемости
                        outfile.write("\n\n")
                except Exception as e:
                    outfile.write(f"Ошибка при чтении файла {filename}: {str(e)}\n\n")
    
    print(f"Все файлы собраны в {output_file}")

# Пример использования
if __name__ == "__main__":
    # Укажите папку, где находятся .py файлы
    source_directory = "./src"  # Текущая папка, можно заменить на другую
    # Укажите имя выходного файла
    output_file = "veector_source_code.txt"
    
    # Запускаем сбор файлов
    collect_python_files_to_txt(source_directory, output_file)
import os

def print_directory_tree(directory, indent=""):
    print(f"{indent}+ {os.path.basename(directory)}/")
    indent += "  "
    for item in sorted(os.listdir(directory)):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            print_directory_tree(path, indent)
        else:
            print(f"{indent}- {item}")

# Укажи корень проекта
root_dir = "/workspaces/Veector"
print_directory_tree(root_dir)
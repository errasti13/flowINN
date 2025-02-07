import ast
import os
import pkg_resources
from pathlib import Path

def get_imports_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read())
        except:
            return set()
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def get_project_imports():
    project_root = Path(__file__).parent.parent
    all_imports = set()
    
    for file_path in project_root.rglob('*.py'):
        if 'venv' not in str(file_path) and 'env' not in str(file_path):
            all_imports.update(get_imports_from_file(str(file_path)))
    
    return all_imports

def get_installed_packages():
    return {pkg.key for pkg in pkg_resources.working_set}

def find_unused_dependencies():
    project_imports = get_project_imports()
    installed_packages = get_installed_packages()
    
    # Always keep these essential packages
    essential_packages = {'pip', 'setuptools', 'wheel'}
    
    unused_packages = installed_packages - project_imports - essential_packages
    return unused_packages

if __name__ == '__main__':
    unused = find_unused_dependencies()
    if unused:
        print("Unused packages found:")
        for pkg in sorted(unused):
            print(f"pip uninstall -y {pkg}")
    else:
        print("No unused packages found!")

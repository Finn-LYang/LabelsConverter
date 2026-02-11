import os
import yaml
from pathlib import Path

def get_files(dir_path, ext=None, recursive=False, target_dir_name=None):
    """
    获取目录下的所有文件
    :param dir_path: 目录路径
    :param ext: 文件扩展名
    :param target_dir_name: 目录路径中包含的目标文件夹名
    :param recursive: 是否递归遍历子文件夹
    :return: 文件列表
    """
    base_path = Path(dir_path)
    if not base_path.is_dir():
        raise NotADirectoryError(f"'{dir_path}' is not a valid directory")

    # 统一扩展名格式：确保是元组且带点，如 ('.jpg', '.png')
    if ext:
        if isinstance(ext, str):
            ext = (ext if ext.startswith('.') else f'.{ext}',)
        else:
            ext = tuple(e if e.startswith('.') else f'.{e}' for e in ext)

    results = []
    
    # 递归逻辑
    if recursive:
        # 使用 rglob 匹配所有文件
        for p in base_path.rglob('*'):
            if p.is_file():
                # 检查是否在目标文件夹名下
                if target_dir_name and target_dir_name not in p.parts:
                    continue
                # 检查后缀
                if not ext or p.suffix.lower() in [e.lower() for e in ext]:
                    results.append(str(p))
    else:
        # 非递归逻辑
        for p in base_path.iterdir():
            if p.is_file():
                if not ext or p.suffix.lower() in [e.lower() for e in ext]:
                    results.append(str(p))
                    
    return results

def load_category_map(yaml_path="./config/category_map.yaml"):
    """
    加载类别映射文件
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Category map not found at: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        try:
            cat_map = yaml.safe_load(f)
            return cat_map
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")
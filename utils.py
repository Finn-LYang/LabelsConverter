import os
import yaml

def get_files(file_dir, ext=None, recursive=False):
    """
    获取目录下的所有文件
    :param file_dir: 目录路径
    :param ext: 文件扩展名
    :param recursive: 是否递归遍历子文件夹
    :return: 文件列表
    """
    if not os.path.isdir(file_dir) and os.path.splitext(file_dir)[1] == ext:
        return [file_dir]
        # raise NotADirectoryError(f"'{file_dir}' is not a valid directory")

    if isinstance(ext, str): # 扩展名是字符串则转换为元组
        ext = (ext,)
        
    results = []

    if recursive:   # 递归遍历子文件夹
        for root, _, files in os.walk(file_dir):
            for file in files:
                if ext is None or file.endswith(ext):
                    results.append(os.path.join(root, file))
    else:
        if ext is None: # 扩展名为空则返回该目录下所有文件
            results = [os.path.join(file_dir, f) for f in os.listdir(file_dir) 
                    if os.path.isfile(os.path.join(file_dir, f))]
        
        else:   
            results = [os.path.join(file_dir, f) for f in os.listdir(file_dir)
                    if os.path.isfile(os.path.join(file_dir, f)) and f.endswith(ext)]

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
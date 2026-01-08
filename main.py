# main.py
import argparse
from core.registry import Registry
from utils import load_category_map
# 引入所有格式插件，触发注册
import formats

def parse_args():
    parser = argparse.ArgumentParser(description="Label Format Converter")
    parser.add_argument("--src-fmt", type=str, default="yolo", help="Source format (coco, labelme)")
    parser.add_argument("--src-label", type=str, default="./_voc2yolo/", help="Path to source data")
    parser.add_argument("--src-image", type=str, default="/path/to/image", help="Path to source data")
    parser.add_argument("--dst-fmt", type=str, default="voc", help="Target format (yolo)")
    parser.add_argument("--dst-path", type=str, default="./_yolo2voc/", help="Path to save output")
    # 新增 config 参数
    parser.add_argument("--category-map", type=str, default="./configs/category_map.yaml", help="Path to category map yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载类别映射
    print(f"Loading category map from {args.category_map}...")
    cat_map = load_category_map(args.category_map)
    print(f"Loaded {len(cat_map)} categories.")

    # 2. 初始化 Reader
    reader_cls = Registry.READERS[args.src_fmt]
    reader = reader_cls(args.src_label, args.src_image, cat_map)

    # 3. 初始化 Writer
    writer_cls = Registry.WRITERS[args.dst_fmt]
    writer = writer_cls(args.dst_path)

    # 4. 执行转换
    print("Starting conversion...")
    writer.write(reader)
    print("Done.")

if __name__ == "__main__":
    main()
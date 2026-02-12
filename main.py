# main.py
import argparse
from pathlib import Path

from core.registry import Registry
from utils import load_category_map
# 引入所有格式插件，触发注册
import formats

def parse_args():
    parser = argparse.ArgumentParser(description="Label Format Converter")
    parser.add_argument("--src-fmt", type=str, default="yolo", help="Source format (coco, labelme)")
    parser.add_argument("--src-label", type=str, default="./test_lab/yolo/labels", help="Path to source data")
    parser.add_argument("--src-image", type=str, default=None, help="Path to source data")
    parser.add_argument("--dst-fmt", type=str, default="coco", help="Target format (yolo)")
    parser.add_argument("--dst-path", type=str, default="./test_lab/yolo/convert_coco/", help="Path to save output")
    # 新增 config 参数
    parser.add_argument("--category-map", type=str, default="./tools/LabelConverter/config/category_map.yaml", help="Path to category map yaml")
    return parser.parse_args()

def main():
    args = parse_args()

    # 0. 字符串处理
    i_lab = Path(args.src_label)
    i_img = Path(args.src_image) if args.src_image is not None else None
    o_lab = Path(args.dst_path)

    # 1. 加载类别映射
    print(f"Loading category map from {args.category_map}...")
    cat_map = load_category_map(args.category_map)
    print(f"Loaded {len(cat_map)} categories.")

    # 2. 初始化 Reader
    reader_cls = Registry.READERS[args.src_fmt]
    reader = reader_cls(i_lab, i_img, cat_map)

    # 3. 初始化 Writer
    writer_cls = Registry.WRITERS[args.dst_fmt]
    writer = writer_cls(o_lab)

    # 4. 执行转换
    print("Starting conversion...")
    writer.write(reader)
    print("Done.")

if __name__ == "__main__":
    main()
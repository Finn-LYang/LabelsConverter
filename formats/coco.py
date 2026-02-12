# COCO JSON 格式的标注文件
# 格式: JSON文件，包含images, annotations, categories字段
# 示例:
# {
#   "images": [{"id": 1, "file_name": "0001.jpg", "width": 640, "height": 480}],
#   "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 150, 200, 250]}],
#   "categories": [{"id": 1, "name": "cat"}]
# }

import json
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Union

from formats.base import BaseReader, BaseWriter
from core.registry import Registry
from core.data_model import BBox, UnifiedLabel
from utils import get_files

@Registry.register_reader("coco")
class CocoReader(BaseReader):
    def __init__(
                self, 
                label_dir: Union[Path, str], 
                image_dir: Optional[Union[Path, str]], 
                categories_map: Optional[Dict[int, str]]
                ):
        self.label_dir = Path(label_dir)
        if image_dir is not None:
            self.image_dir = Path(image_dir)
        else:
            self.image_dir = label_dir.parent / 'images'
        self.categories_map = categories_map

        # 获取目录下所有的 JSON 文件
        if self.label_dir.is_file() and self.label_dir.suffix == '.json':
            self.files = [self.label_dir]
        else:
            self.files = get_files(self.label_dir, ext='.json')

        # 建立索引
        self.samples = []
        self._build_index()
    
    def __len__(self):
        return len(self.samples)

    def _build_index(self):
        """ 预加载所有 JSON 文件，并建立图像、类别和标注的索引 """
        print(f"[CocoReader] Loading annotations from {len(self.files)} files...")

        for json_path in self.files:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 类别ID -> 类别名
            local_cat_map = {cat['id']:cat['name'] for cat in data.get('categories', [])}
            # 图片ID -> 标注信息(id, image_id, category_id, bbox)
            img_to_anns = defaultdict(list)
            if 'annotations' in data:
                for ann in data['annotations']:
                    img_to_anns[ann['image_id']].append(ann)

            # 遍历 Images 构建样本
            images = data.get('images', [])
            for img_info in images:
                file_name = img_info['file_name']
                # 处理相对路径或绝对路径
                image_path = self.image_dir / file_name
                
                img_id = img_info['id']
                raw_anns = img_to_anns.get(img_id, [])
                
                parsed_bboxes = []
                for ann in raw_anns:
                    # COCO bbox format: [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    cat_id = ann['category_id']
                    
                    # 转换坐标为 xmin, ymin, xmax, ymax
                    xmin, ymin = x, y
                    xmax, ymax = x + w, y + h
                    
                    # 获取类别名称
                    label_name = local_cat_map.get(cat_id, str(cat_id))
                    
                    # 如果有全局映射限制，可以在这里做过滤或转换
                    # 此处直接构建 BBox
                    parsed_bboxes.append(
                        BBox(
                            xmin=float(xmin), 
                            ymin=float(ymin), 
                            xmax=float(xmax), 
                            ymax=float(ymax), 
                            cls_id=cat_id, 
                            label_name=label_name
                        )
                    )
                
                # 将该图片的信息存入 samples 列表
                self.samples.append({
                    "image_path": image_path,
                    "width": img_info['width'],
                    "height": img_info['height'],
                    "bboxes": parsed_bboxes
                })
        
        print(f"[CocoReader] Loaded {len(self.samples)} images.")

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return UnifiedLabel(
                    image_path=sample['image_path'],
                    image_width=sample['width'],
                    image_height=sample['height'],
                    bboxes=sample['bboxes'],
                    masks=[]
                )


@Registry.register_writer("coco")
class CocoWriter(BaseWriter):
    def __init__(self, output_path: Path):
        super().__init__(output_path)
        if output_path.suffix == '.json':
            self.annotation_path = output_path
            self.output_dir = output_path.parent
        else:
            self.output_dir = output_path
            self.annotation_path = output_path / "instances_default.json"
            
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
    def write(self, labels: Iterator[UnifiedLabel]):
        """
        转换成COCO格式的标注文件
        labels: Iterator[UnifiedLabel]
        """
        images = []
        annotations = []
        categories = []

        # 动态映射，未知的新类别
        category_name_to_id = {}

        # 计数器
        ann_id_counter = 1
        
        # 逐个图像生成COCO格式数据
        label_list = list(labels)
        for img_id, label in tqdm(enumerate(label_list, start=1), total=len(label_list), desc="Exporting COCO"):
            # 1. 构建 Image 信息
            # 如果 label 中没有文件名，生成一个数字文件名
            if label.image_path:
                file_name = label.image_path.name
            else:
                file_name = f"{img_id:06d}.jpg"
                
            images.append({
                "id": img_id,
                "file_name": file_name,
                "width": int(label.image_width),
                "height": int(label.image_height)
            })
            
            # 2. 构建 Annotation 信息
            for bbox in label.bboxes:
                label_name = bbox.label_name if bbox.label_name else "unknown"
                
                # 维护类别 ID
                if label_name not in category_name_to_id:
                    new_id = len(category_name_to_id) + 1
                    category_name_to_id[label_name] = new_id
                    categories.append({
                        "id": new_id,
                        "name": label_name,
                        "supercategory": "none"
                    })
                
                category_id = category_name_to_id[label_name]
                
                # 坐标转换：xyxy -> xywh
                # 确保宽度和高度非负
                w = max(0.0, float(bbox.xmax - bbox.xmin))
                h = max(0.0, float(bbox.ymax - bbox.ymin))
                x = float(bbox.xmin)
                y = float(bbox.ymin)
                
                annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h, # COCO 标准通常包含 area
                    "iscrowd": 0,
                    "segmentation": [] # 仅检测框时为空列表
                })
                ann_id_counter += 1
        
        # 3. 组装最终字典
        coco_dict = {
            "info": {
                "description": "Converted by CocoWriter",
                "year": 2024,
                "version": "1.0"
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        # 4. 写入文件
        print(f"[CocoWriter] Saving to {self.annotation_path} ...")
        with open(str(self.annotation_path), "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False, indent=2)
        
        return self.annotation_path
# COCO JSON 格式的标注文件
# 格式: JSON文件，包含images, annotations, categories字段
# 示例:
# {
#   "images": [{"id": 1, "file_name": "0001.jpg", "width": 640, "height": 480}],
#   "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 150, 200, 250]}],
#   "categories": [{"id": 1, "name": "cat"}]
# }

import json
import os
from tqdm import tqdm
from collections import defaultdict
from formats.base import BaseReader, BaseWriter
from core.registry import Registry
from core.data_model import BBox, UnifiedLabel
from utils import get_files

@Registry.register_reader("coco")
class CocoReader(BaseReader):
    def __init__(self, label_dir, image_dir, categories_map):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.files = get_files(self.label_dir, '.json')
        self.categories_map = categories_map
    
    def __len__(self):
        return len(self.files)

    def _create_index(self, json_path):
        """ 创建图像、类别和标注的索引 """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 类别ID -> 类别名
        idx_to_name = {cat['id']:cat['name'] for cat in data.get('categories', [])}
        # 图片ID -> 图片信息
        images_info = {img['id']: img for img in data.get('images', [])}
        # 图片ID -> 标注信息(id, image_id, category_id, bbox)
        img_to_anns = defaultdict(list)
        if 'annotations' in data:
            for ann in data['annotations']:
                img_to_anns[ann['image_id']].append(ann)
        
        return idx_to_name, images_info, img_to_anns
        
    def __iter__(self):
        for file in self.files:
            idx_to_name, images_info, img_to_anns = self._create_index(file)
            
            # 逐图像解析
            for img_id, img_info in images_info.items():
                file_name = img_info['file_name']
                image_path = os.path.join(self.image_dir, file_name)
                w, h = img_info['width'], img_info['height']
                
                anns = img_to_anns[img_id]
                bboxes = []
                for ann in anns:
                    category_id = ann['category_id']
                    x, y, bw, bh = ann['bbox']
                    xmin = x
                    ymin = y
                    xmax = x + bw
                    ymax = y + bh
                    label_name = idx_to_name.get(category_id, str(category_id))
                    bboxes.append(BBox(xmin, ymin, xmax, ymax, cls_id=category_id, label_name=label_name))
                
                # 每一张图像生成一个UnifiedLabel
                yield UnifiedLabel(image_path=image_path, image_width=w, image_height=h, bboxes=bboxes, masks=[])


@Registry.register_writer("coco")
class CocoWriter(BaseWriter):
    def __init__(self, output_path):
        super().__init__(output_path)
        if os.path.isfile(output_path):
            self.annotation_path = output_path
        else:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.annotation_path = os.path.join(output_path, "annotations.json")
            
    def write(self, labels):
        """
        转换成COCO格式的标注文件
        labels: Iterator[UnifiedLabel]
        """
        images = []
        annotations = []
        categories = []
        category_name_to_id = {}
        ann_id = 1
        
        # 逐个图像生成COCO格式数据
        labels = list(labels)
        for img_id, label in tqdm(enumerate(labels, start=1), total=len(labels), desc="Converting to COCO format"):
            file_name = os.path.basename(label.image_path) if label.image_path else f"{img_id:06d}.jpg"
            images.append({
                "id": img_id,
                "file_name": file_name,
                "width": int(label.image_width),
                "height": int(label.image_height)
            })
            
            # 处理当前图像中的每个边界框
            for bbox in label.bboxes:
                label_name = bbox.label_name or "unknown"
                if label_name not in category_name_to_id:
                    category_id = len(category_name_to_id) + 1
                    category_name_to_id[label_name] = category_id
                    categories.append({
                        "id": category_id,
                        "name": label_name
                    })
                else:
                    category_id = category_name_to_id[label_name]
                
                x = float(bbox.xmin)
                y = float(bbox.ymin)
                width = float(bbox.xmax - bbox.xmin)
                height = float(bbox.ymax - bbox.ymin)
                
                # 添加边界框注释
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x, y, width, height]
                })
                ann_id += 1
        
        # 构建COCO格式字典
        coco_dict = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        with open(self.annotation_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False, indent=2)
        
        return self.annotation_path

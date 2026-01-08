# Yolo txt 格式的标注文件
# 格式: class_id x_center y_center width height (normalized)

import os
import cv2
from tqdm import tqdm
from formats.base import BaseReader, BaseWriter
from core.registry import Registry
from core.data_model import BBox, UnifiedLabel
from utils import get_files

@Registry.register_reader("yolo")
class YoloReader(BaseReader):
    def __init__(self, label_dir, image_dir, categories_map):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.files = get_files(label_dir, '.txt')
        self.categories_map = categories_map
    
    def __len__(self):
        return len(self.files)

    # def _load_class_names(self):
    #     """ 从yaml文件中加载类别名称 """
    #     if not self.yaml_path or not os.path.exists(self.yaml_path):
    #         return {}
    #     with open(self.yaml_path, 'r') as f:
    #         data = yaml.safe_load(f)
    #         names = data.get('names', {})
    #         if isinstance(names, list):
    #             # 转换为字典格式 {0: 'class0', 1: 'class1', ...}
    #             return {i: name for i, name in enumerate(names)}
    #         return names

    def __iter__(self):
        for file in self.files:
            # 获取对应图像的地址和信息
            img_path = os.path.join(self.image_dir, os.path.splitext(os.path.basename(file))[0] + '.jpg')
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            bboxes = []
            with open(file, 'r') as f:
                for line in f:
                    cls_id, cx, cy, bw, bh = map(float, line.strip().split())
                    # 在这里进行坐标转换：归一化 cxcywh -> 绝对 xyxy
                    xmin = (cx - bw / 2) * w
                    ymin = (cy - bh / 2) * h
                    xmax = (cx + bw / 2) * w
                    ymax = (cy + bh / 2) * h

                    label_name = self.categories_map.get(int(cls_id), str(int(cls_id)))
                    bboxes.append(BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, cls_id=int(cls_id), label_name=label_name))

            yield UnifiedLabel(image_path=img_path, image_width=w, image_height=h, bboxes=bboxes, masks=[])

@Registry.register_writer("yolo")
class YoloWriter(BaseWriter):
    def __init__(self, output_path):
        super().__init__(output_path)
        self.output_dir = output_path
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
    
    def write(self, labels):
        """
        将UnifiedLabel列表转换为YOLO格式的标注文件
        labels: Iterator[UnifiedLabel]
        """
        # 逐个图像生成YOLO格式的标注文件
        labels = list(labels)
        for label in tqdm(labels, total=len(labels), desc="Converting to YOLO format"):
            file_name = os.path.basename(label.image_path)
            txt_file = os.path.splitext(file_name)[0] + '.txt'
            save_path = os.path.join(self.output_dir, txt_file)
            
            # 写入YOLO格式的标注
            with open(save_path, 'w') as f:
                for bbox in label.bboxes:
                    # 转换为YOLO格式：normalized cxcywh
                    x_center = (bbox.xmin + bbox.xmax) / 2 / label.image_width
                    y_center = (bbox.ymin + bbox.ymax) / 2 / label.image_height
                    width = (bbox.xmax - bbox.xmin) / label.image_width
                    height = (bbox.ymax - bbox.ymin) / label.image_height
                    
                    f.write(f"{bbox.cls_id} {x_center} {y_center} {width} {height}\n")

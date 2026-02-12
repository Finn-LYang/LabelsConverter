# Yolo txt 格式的标注文件
# 格式: class_id x_center y_center width height (normalized)

import os
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Union

from formats.base import BaseReader, BaseWriter
from core.registry import Registry
from core.data_model import BBox, UnifiedLabel
from utils import get_files

@Registry.register_reader("yolo")
class YoloReader(BaseReader):
    def __init__(
            self, 
            label_dir: Union[Path, str], 
            image_dir: Optional[Union[Path, str]], 
            categories_map: Optional[Dict[int, str]]
            ):
        self.label_dir = label_dir
        if image_dir is not None:
            self.image_dir = image_dir
        else:
            self.image_dir = label_dir.parent / 'images'
        self.files = get_files(label_dir, '.txt')
        self.categories_map = categories_map
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        """
        根据索引获取单个标注数据
        """
        if idx >= len(self.files):
            raise IndexError(f"Index {idx} out of range for {len(self.files)} files")
        
        file = self.files[idx]
        file = Path(file)
        return self._process(file)
        

    def _process(self, file: Path):
        """
        处理单个 YOLO 标注文件
        :param file: 标注文件路径
        :return: UnifiedLabel对象或None(如果图像不存在)
        """
        # 获取对应图像的地址和信息
        img_path = self.image_dir / (file.stem + '.jpg')
        
        # 判断图像文件是否存在
        if not img_path.exists():
            return None
        
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w = img.shape[:2]
    
        bboxes = []
        with open(file, 'r') as f:
            for line in f:
                cls_id, cx, cy, bw, bh = map(float, line.strip().split())
                # 坐标转换：归一化 cxcywh -> 绝对 xyxy
                xmin = (cx - bw / 2) * w
                ymin = (cy - bh / 2) * h
                xmax = (cx + bw / 2) * w
                ymax = (cy + bh / 2) * h
    
                label_name = self.categories_map.get(int(cls_id), str(int(cls_id)))
                bboxes.append(BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, cls_id=int(cls_id), label_name=label_name))
    
        return UnifiedLabel(image_path=img_path, image_width=w, image_height=h, bboxes=bboxes, masks=[])
        
@Registry.register_writer("yolo")
class YoloWriter(BaseWriter):
    def __init__(self, output_path: Path):
        super().__init__(output_path)
        self.output_dir = output_path
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write(self, labels):
        """
        将UnifiedLabel列表转换为YOLO格式的标注文件
        labels: Iterator[UnifiedLabel]
        """
        # 逐个图像生成YOLO格式的标注文件
        labels = list(labels)
        for label in tqdm(labels, total=len(labels), desc="Converting to YOLO format"):
            file_name = label.image_path.stem
            txt_file = file_name + '.txt'
            save_path = self.output_dir / txt_file
            
            # 写入YOLO格式的标注
            with open(str(save_path), 'w') as f:
                for bbox in label.bboxes:
                    # 转换为YOLO格式：normalized cxcywh
                    x_center = (bbox.xmin + bbox.xmax) / 2 / label.image_width
                    y_center = (bbox.ymin + bbox.ymax) / 2 / label.image_height
                    width = (bbox.xmax - bbox.xmin) / label.image_width
                    height = (bbox.ymax - bbox.ymin) / label.image_height
                    
                    f.write(f"{bbox.cls_id} {x_center} {y_center} {width} {height}\n")

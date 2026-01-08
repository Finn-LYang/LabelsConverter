# 统一标签数据模型
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class BBox:
    """ 统一使用 xyxy 格式存储绝对坐标 """
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    cls_id: int
    label_name: str

    @property
    def width(self): return self.xmax - self.xmin
    @property
    def height(self): return self.ymax - self.ymin

@dataclass
class UnifiedLabel:
    """ 
    统一标签数据结构，
    可用此结构转换成任意标签格式。
    """
    image_path: str
    image_width: int
    image_height: int
    bboxes: List[BBox]
    masks: List[np.ndarray] # Optional: 用于分割
    
    class Config:
        arbitrary_types_allowed = True
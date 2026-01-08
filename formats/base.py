# 定义抽象的基类

from abc import ABC, abstractmethod
from typing import Iterator
from core.data_model import UnifiedLabel

class BaseReader(ABC):  # 不可实例化，只能被继承
    def __init__(self, data_path: str, **kwargs):
        self.data_path = data_path
    
    def __len__(self):
        return len(self.data_path)

    @abstractmethod
    def __iter__(self) -> Iterator[UnifiedLabel]:
        """生成统一的数据模型"""
        pass

class BaseWriter(ABC):
    def __init__(self, output_path: str, **kwargs):
        self.output_path = output_path

    @abstractmethod
    def write(self, labels: Iterator[UnifiedLabel]):
        """接收统一数据模型流，写入文件"""
        pass
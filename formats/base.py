# 定义抽象的基类

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union
from core.data_model import UnifiedLabel

class BaseReader(ABC):  # 不可实例化，只能被继承
    def __init__(self, data_path: Union[Path, str], **kwargs):
        self.data_path = Path(data_path)
    
    @abstractmethod
    def __len__(self):
        pass 

    @abstractmethod
    def __getitem__(self, idx: int) -> UnifiedLabel:
        pass

class BaseWriter(ABC):
    def __init__(self, output_path: Union[Path, str], **kwargs):
        self.output_path = Path(output_path)

    @abstractmethod
    def write(self, labels: Iterator[UnifiedLabel]):
        """接收统一数据模型流，写入文件"""
        pass
# VOC XML 格式的标注文件
# 格式: <folder>/<filename>.jpg, <object>标签包含<name>, <bndbox>(xmin,ymin,xmax,ymax)
# 示例:
# <annotation>
#   <folder>images</folder>
#   <filename>0001.jpg</filename>
#   <object>
#     <name>cat</name>
#     <bndbox>
#       <xmin>100</xmin>
#       <ymin>150</ymin>
#       <xmax>300</xmax>
#       <ymax>400</ymax>
#     </bndbox>
#   </object>
# </annotation>

import os
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom
from formats.base import BaseReader, BaseWriter
from core.registry import Registry
from core.data_model import BBox, UnifiedLabel
from utils import get_files

@Registry.register_reader("voc")
class VocReader(BaseReader):
    def __init__(self, label_dir, image_dir, categories_map):
        self.input_dir = label_dir
        self.image_dir = image_dir
        self.files = get_files(label_dir, '.xml')
        # self.categories_map = categories_map
        self.categories_map = {v: k for k, v in categories_map.items()}

    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        for file in self.files:
            # 解析XML文件
            tree = ET.parse(file)
            root = tree.getroot()
            
            # 获取图片信息
            # folder = root.find('folder').text
            filename = root.find('filename').text
            img_path = os.path.join(self.image_dir, filename)
            img_size = root.find('size')
            if img_size is not None:
                w = int(img_size.find('width').text)
                h = int(img_size.find('height').text)
            else:
                img = cv2.imread(img_path)
                h, w, _ = img.shape
            
            bboxes = []
            # 解析object标签中的bbox信息
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                cls_id = self.categories_map.get(cls_name, -1)
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # 构造UnifiedLabel对象
                bboxes.append(BBox(xmin, ymin, xmax, ymax, cls_id=cls_id, label_name=cls_name))
            
            yield UnifiedLabel(image_path=img_path, image_width=w, image_height=h, bboxes=bboxes, masks=[])

@Registry.register_writer("voc")
class VocWriter(BaseWriter):
    def __init__(self, output_path):
        super().__init__(output_path)
        self.output_dir = output_path
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _prettify(self, elem):
        """将 ElementTree 转换为美化的字符串格式"""
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def write(self, labels):
        """
        将UnifiedLabel列表转换为VOC格式的标注文件
        labels: Iterator[UnifiedLabel]
        """
        # 逐个图像生成VOC格式的标注文件
        labels = list(labels)
        for label in tqdm(labels, total=len(labels), desc="Converting to VOC format"):
            file_name = os.path.basename(label.image_path)
            xml_file = os.path.splitext(file_name)[0] + '.xml'
            save_path = os.path.join(self.output_dir, xml_file)

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = os.path.basename(os.path.dirname(label.image_path))
            ET.SubElement(root, "filename").text = file_name
            ET.SubElement(root, "path").text = label.image_path

            for bbox in label.bboxes:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = str(bbox.label_name)

                # 构造 Bndbox
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(bbox.xmin))
                ET.SubElement(bndbox, "ymin").text = str(int(bbox.ymin))
                ET.SubElement(bndbox, "xmax").text = str(int(bbox.xmax))
                ET.SubElement(bndbox, "ymax").text = str(int(bbox.ymax))

            xml_str = self._prettify(root)
            with open(save_path, "w", encoding='utf-8') as f:
                lines = xml_str.split('\n')
                f.write('\n'.join(lines[1:]) if lines[0].startswith('<?xml') else xml_str)


# 代码结构
label_converter/
├── configs/                 # 配置文件
│   ├── category_map.yaml    # 类别映射表 (e.g., "person" -> 0)
│   └── default.yaml         # 全局配置
├── core/                    # 核心逻辑
│   ├── __init__.py
│   ├── data_model.py        # 定义 UnifiedLabel, BBox 等数据结构 (IR)
│   └── registry.py          # 注册机制，用于管理所有的 Reader/Writer
├── formats/                 # 各个格式的具体实现 (插件)
│   ├── __init__.py
│   ├── base.py              # 定义 BaseReader 和 BaseWriter 抽象基类
│   ├── coco.py              # COCO 格式的 Reader/Writer 实现
│   ├── yolo.py              # YOLO 格式的 Reader/Writer 实现
│   ├── voc.py               # VOC XML 格式的 Reader/Writer 实现
│   └── labelme.py           # LabelMe JSON 格式的 Reader/Writer 实现
├── utils/                   # 工具类
│   ├── image_utils.py       # 读取图片宽高 (Lazy loading)
│   ├── path_utils.py        # 路径处理
│   └── vis_utils.py         # 转换后的可视化验证工具
├── tests/                   # 单元测试 (非常重要，尤其是坐标转换)
├── main.py                  # 统一入口
├── requirements.txt
└── README.md

# 使用方法 
python main.py --src-fmt yolo --src-label /path/to/input --src-image /path/to/image --dst-fmt voc --dst-path /path/to/output --category-map /path/to/category_map.yaml
# 读写注册机制

class Registry:
    """
    用于注册和管理各种格式的读写器
    """
    READERS = {}
    WRITERS = {}

    @classmethod
    def register_reader(cls, name):
        """注册读取器"""
        def decorator(obj):
            cls.READERS[name] = obj
            return obj
        return decorator

    @classmethod
    def register_writer(cls, name):
        """注册写入器"""
        def decorator(obj):
            cls.WRITERS[name] = obj
            return obj
        return decorator
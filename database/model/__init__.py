from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import DeclarativeMeta
# ORM基类
Base: DeclarativeMeta = declarative_base()
# 导入orm，注意依赖顺序
from . import user
from . import file
from . import knowledge_base
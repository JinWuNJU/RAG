from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import DeclarativeMeta
# ORM基类
Base: DeclarativeMeta = declarative_base()

from . import user
from . import file
from . import knowledge_base
from . import evaluation
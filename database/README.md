### 数据库相关
#### `database/__init__.py`文件
创建数据库引擎，建立ORM对应的数据表（通过导入`database/model/__init__.py`并调用`create_all()`）
提供FastAPI依赖get_db()，用于获取数据库会话

#### model文件夹
存放ORMs，其中`database/model/__init__.py`用于提供ORM基类，如果定义了新的ORM，需要在该文件内导入，使得db_engine能够创建对应的表
`database/model/__init__.py`
```py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import DeclarativeMeta
# ORM基类
Base: DeclarativeMeta = declarative_base()

from . import user
from . import file
from . import knowledge_base
from . import evaluation
# ...添加其他ORM
```
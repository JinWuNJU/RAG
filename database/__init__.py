import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 加载环境变量
load_dotenv()

# 构建连接URL
def get_db_url():
    return f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@" \
           f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}" \
           f"?sslmode={os.getenv('DB_SSL_MODE', 'prefer')}"

# 创建引擎（连接池配置）
engine = create_engine(
    get_db_url(),
    pool_size=10,          # 连接池保持的连接数
    max_overflow=20,       # 超出pool_size时允许的最大连接数
    pool_pre_ping=True,    # 自动检测连接是否有效
    pool_recycle=3600,     # 1小时后回收连接（避免数据库超时）
    connect_args={
        "connect_timeout": 5,  # 连接超时5秒
        "application_name": "RAG_APP"  # 在PG中标识连接来源
    }
)

# 建表
from database.model import Base
Base.metadata.create_all(bind=engine)

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from contextlib import contextmanager

@contextmanager
def get_db_with():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
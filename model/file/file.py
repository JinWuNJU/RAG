from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, LargeBinary, DateTime, Boolean, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class FileDB(Base):
    """数据库文件存储模型"""
    __tablename__ = "files"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    data = Column(LargeBinary, nullable=False)
    user_id = Column(PGUUID(as_uuid=True), nullable=True)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FileMetadata(BaseModel):
    """文件元数据模型"""
    id: UUID
    filename: str
    content_type: str
    size: int
    is_public: bool
    created_at: datetime
    updated_at: datetime
    
class FileUploadResponse(BaseModel):
    """文件上传响应模型"""
    file_id: UUID
    
class FileTypeError(Exception):
    """文件类型错误"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
        
class FileSizeError(Exception):
    """文件大小错误"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message) 
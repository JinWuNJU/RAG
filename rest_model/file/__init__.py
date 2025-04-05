from uuid import UUID
from pydantic import BaseModel, Field
from datetime import datetime
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
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import UUID


class KnowledgeBaseCreate(BaseModel):
    """创建知识库请求模型"""
    name: str
    description: Optional[str] = None
    file_ids: List[UUID]
    chunk_size: int = 1024
    overlap_size: int = 256
    hybrid_ratio: float = 0.5

class KnowledgeBaseCreateResponse(BaseModel):
    """知识库创建响应模型"""
    knowledge_base_id: UUID
    status: str

class KnowledgeBaseListItem(BaseModel):
    """知识库列表项响应模型"""
    knowledge_base_id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    status: str  # "building" 或 "completed"

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()  # 确保datetime序列化
        }

class KnowledgeBaseDetailResponse(BaseModel):
    """知识库列表项响应模型"""
    knowledge_base_id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    status: str  # "building" 或 "completed"
    chunk_size: int
    overlap_size: int
    hybrid_ratio: float

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()  # 确保datetime序列化
        }

class KnowledgeBaseSearchResult(BaseModel):
    """知识库搜索结果模型"""
    content: str
    file_name: str
    file_id: UUID
    chunk_index: int
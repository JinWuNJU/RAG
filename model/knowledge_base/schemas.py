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

class KnowledgeBaseResponse(BaseModel):
    """知识库响应模型"""
    knowledge_base_id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    status: str


class KnowledgeBaseSearchResult(BaseModel):
    """知识库搜索结果模型"""
    content: str
    file_name: str
    file_id: UUID
    chunk_index: int
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
    is_public: bool = False

class KnowledgeBaseCreateResponse(BaseModel):
    """知识库创建响应模型"""
    knowledge_base_id: UUID
    status: str

class KnowledgeBaseBasicInfo(BaseModel):
    knowledge_base_id: UUID
    name: str
    description: Optional[str]

class KnowledgeBaseListItem(KnowledgeBaseBasicInfo):
    """知识库列表项响应模型"""
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

class SearchRequest(BaseModel):
    """搜索请求体"""
    query: str
    limit: int = 10  # 默认返回10条结果

class SearchResult(BaseModel):
    content: str
    file_id: UUID
    chunk_index: int
    
    class Config:
        json_encoders = {
            UUID: lambda v: str(v)  # UUID转为字符串
        }
    """搜索结果响应模型"""
    file_name: str    # 直接从knowledge_base_chunks表获取

class SearchScoreResult(SearchResult):
    score: float
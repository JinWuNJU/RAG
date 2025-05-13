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

class KnowledgeBaseListRequest(BaseModel):
    """知识库列表请求模型"""
    name: Optional[str] = None
    page: int = 0
    limit: int = 10

class KnowledgeBaseBasicInfo(BaseModel):
    knowledge_base_id: UUID
    name: str
    description: Optional[str]



class KnowledgeBaseListItem(KnowledgeBaseBasicInfo):
    """知识库列表项响应模型"""
    created_at: datetime
    status: str  # "building" 或 "completed"
    uploader_id:UUID
    is_public: bool

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()  # 确保datetime序列化
        }

class PaginatedResponse(BaseModel):
    items: List[KnowledgeBaseListItem]
    total: int
    page: int
    limit: int
    total_pages: int

class KnowledgeBaseFile(BaseModel):
    """知识库文件响应模型"""
    file_id: UUID
    file_name: str
    file_size: int
    chunk_count: int
    created_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),  # 确保datetime序列化
            UUID: lambda v: str(v)  # UUID转为字符串
        }

class KnowledgeBaseDetailResponse(BaseModel):
    """知识库列表项响应模型"""
    knowledge_base_id: UUID
    name: str
    description: Optional[str]
    created_at: datetime
    status: str  # "building" 或 "completed"
    uploader_id: UUID
    is_public: bool
    chunk_size: int
    overlap_size: int
    hybrid_ratio: float
    files: List[KnowledgeBaseFile] = []

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
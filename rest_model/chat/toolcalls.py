from typing import List

from pydantic import BaseModel, Field


class RetrieveParams(BaseModel):
    """检索工具的参数"""
    knowledge_base: str = Field(..., description="知识库名称")
    keyword: str = Field(..., description="检索关键词")

class RetrievedDocument(BaseModel):
    """检索到的文档信息"""
    snippet: str = Field(..., description="文档片段")
    url: str = Field(..., description="文档链接")

class RetrieveToolReturn(BaseModel):
    """返回文档事件"""
    count: int = Field(..., description="文档数量")
    documents: List[RetrievedDocument] = Field(..., description="文档列表")
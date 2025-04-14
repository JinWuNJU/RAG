from abc import ABC
from typing import List, Literal, Annotated
from uuid import UUID

from pydantic import BaseModel, Field

class BaseEvent(BaseModel):
    """基础事件模型"""
    type: str = Field(..., description="事件类型")

class RetrieveParams(BaseModel):
    """检索工具的参数"""
    knowledge_base: str = Field(..., description="知识库名称")
    keyword: str = Field(..., description="检索关键词")

class Document(BaseModel):
    """检索到的文档信息"""
    snippet: str = Field(..., description="文档片段")
    url: str = Field(..., description="文档链接")

class ToolCallEvent(BaseEvent):
    """工具调用事件"""
    type: Annotated[str, Literal["call"]] = "call"
    name: str = Field(..., description="工具名称")
    params: RetrieveParams | str | dict = Field(..., description="工具参数")
    description: str = Field(..., description="调用描述")

class DocReturnEvent(BaseEvent):
    """返回文档事件"""
    type: Annotated[str, Literal["doc"]] = "doc"
    count: int = Field(..., description="文档数量")
    documents: List[Document] = Field(..., description="文档列表")

class ChatEvent(BaseEvent):
    """聊天事件"""
    type: Annotated[str, Literal["chat"]] = "chat"
    content: str = Field(..., description="聊天内容")

class ChatBeginEvent(BaseEvent):
    """SSE开始事件"""
    type: Annotated[str, Literal["begin"]] = "begin"
    chat_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID


ToolEvent = ToolCallEvent | DocReturnEvent
SseEvent = ToolEvent | ChatEvent | ChatBeginEvent

def SseEventPackage(event: SseEvent) -> dict:
    return {
        "event": event.type,
        "data": event.model_dump_json()
    }
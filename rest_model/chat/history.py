from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from rest_model.chat.sse import ToolEvent


class ChatMessage(BaseModel):
    """聊天消息模型"""
    parentId: Optional[UUID]
    id: UUID
    role: str
    content: str
    tooluse: Optional[List[ToolEvent]] = None
    timestamp: int

class ChatDetail(BaseModel):
    """单条聊天历史的对话详情"""
    messages: List[ChatMessage] = Field(..., description="聊天消息列表")

class ChatHistory(BaseModel):
    """聊天历史列表项"""
    id: UUID
    title: str
    chat: ChatDetail
    updated_at: int
    created_at: int


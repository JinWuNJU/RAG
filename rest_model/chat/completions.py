from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class MessagePayload(BaseModel):
    """用于接收用户消息的模型"""
    content: str
    parentId: Optional[str] = None
    chatId: Optional[str] = None
    knowledgeBase: Optional[List[UUID]] = None # 当前提问时，向ai提供的知识库，None表示不使用知识库
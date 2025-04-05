from typing import Optional

from pydantic import BaseModel


class MessagePayload(BaseModel):
    """用于接收用户消息的模型"""
    content: str
    parentId: Optional[str] = None
    chatId: Optional[str] = None
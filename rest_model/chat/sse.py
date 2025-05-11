from typing import Literal, Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from rest_model.chat.history import ChatToolCallPart, ChatToolReturnPart


class BaseEvent(BaseModel):
    """基础事件模型"""
    type: str = Field(..., description="事件类型")

class ToolCallEvent(BaseEvent):
    """工具调用事件"""
    type: Annotated[str, Literal["tool-call"]] = "tool-call"
    data: ChatToolCallPart

class ToolReturnEvent(BaseEvent):
    type: Annotated[str, Literal["tool-return"]] = "tool-return"
    data: ChatToolReturnPart

class ChatEvent(BaseEvent):
    """聊天事件"""
    type: Annotated[str, Literal["chat"]] = "chat"
    content: str = Field(..., description="聊天内容")

class ChatBeginEvent(BaseEvent):
    """流式对话开始事件"""
    type: Annotated[str, Literal["begin"]] = "begin"
    chat_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    
class ChatEndEvent(BaseEvent):
    """流式对话结束事件"""
    type: Annotated[str, Literal["end"]] = "end"
    
class ChatTitleEvent(BaseEvent):
    """
    对话标题事件
    在ChatEndEvent后发送
    标记SSE流结束
    """
    type: Annotated[str, Literal["title"]] = "title"
    title: str = Field(..., description="对话标题")


ToolEvent = ToolCallEvent | ToolReturnEvent
SseEvent = ToolEvent | ChatEvent | ChatBeginEvent | ChatEndEvent | ChatTitleEvent

def SseEventPackage(event: SseEvent) -> dict:
    return {
        "event": event.type,
        "data": event.model_dump_json()
    }
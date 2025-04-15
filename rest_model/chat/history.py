from typing import TYPE_CHECKING, Annotated, Any, List, Literal, Optional, Union
from uuid import UUID

import pydantic
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from database.model.chat import ChatMessageDB, ChatHistoryDB

# === 移除了不需要成员的pydantic_ai.messagesModelMessage ===

class ChatTextPart(BaseModel):
    part_kind: Literal["text"] = "text"
    content: str

class ChatToolCallPart(BaseModel):
    part_kind: Literal["tool-call"] = "tool-call"
    tool_name: str
    args: str | dict[str, Any]

class ChatToolReturnPart(BaseModel):
    part_kind: Literal["tool-return"] = "tool-return"
    tool_name: str
    content: str | dict[str, Any]

ChatMessagePart = Annotated[Union[ChatTextPart, ChatToolCallPart, ChatToolReturnPart], pydantic.Discriminator('part_kind')]
# ============

class ChatMessage(BaseModel):
    """聊天消息模型"""
    parentId: Optional[UUID]
    id: UUID
    role: Literal["user", "assistant"]
    part: List[ChatMessagePart] = Field(default_factory=list)
    timestamp: int

    @classmethod
    def from_orm(cls, obj: "ChatMessageDB"):
        part = []
        for msg in obj.part:
            if msg.kind == "request":
                for p in msg.parts:
                    if p.part_kind == "user-prompt":
                        part.append(ChatTextPart(content=str(p.content)))
                    elif p.part_kind == "tool-return":
                        part.append(ChatToolReturnPart(content=p.content, tool_name=p.tool_name))
            elif msg.kind == "response":
                for p in msg.parts:
                    if p.part_kind == "text":
                        part.append(ChatTextPart(content=str(p.content)))
                    elif p.part_kind == "tool-call":
                        part.append(ChatToolCallPart(tool_name=p.tool_name, args=p.args))
        return cls(
            parentId=obj.parent_id,
            id=obj.id,
            role=obj.role,
            part=part,
            timestamp=int(obj.timestamp.timestamp())
        )

class ChatHistory(BaseModel):
    """聊天历史列表项"""
    id: UUID
    title: str
    updated_at: int
    created_at: int

    @classmethod
    def from_orm(cls, obj: "ChatHistoryDB"):
        return cls(
            id=obj.id,
            title=obj.title,
            updated_at=int(obj.updated_at.timestamp()),
            created_at=int(obj.created_at.timestamp())
        )

class ChatDialog(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)

class ChatDetail(ChatHistory):
    """单条聊天历史的对话详情"""
    chat: ChatDialog

    @classmethod
    def from_orm(cls, obj: "ChatHistoryDB"):
        super_model = ChatHistory.from_orm(obj)
        return cls(
            **super_model.model_dump(),
            chat=ChatDialog(messages=[ChatMessage.from_orm(m) for m in obj.chat])
        )
    
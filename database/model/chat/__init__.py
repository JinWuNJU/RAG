import json
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Literal
from uuid import UUID

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_core import to_jsonable_python
from sqlalchemy import ForeignKey, Index, String, TIMESTAMP, TypeDecorator, asc, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.model import Base

if TYPE_CHECKING:
    pass
    
class JSONEncodedModelMessage(TypeDecorator):
    impl = String

    def process_bind_param(self, value, dialect):
        if isinstance(value, list):
            return json.dumps(to_jsonable_python(value), ensure_ascii=False)
        return None

    def process_result_value(self, value, dialect):
        if isinstance(value, str):
            return ModelMessagesTypeAdapter.validate_json(value)
        return None

class ChatMessageDB(Base):
    """数据库聊天消息存储模型"""
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("chat_messages.id"), nullable=True)
    chat_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("chat_history.id"), nullable=False)
    role: Mapped[Literal["user", "assistant"]] = mapped_column(String, nullable=False)
    part: Mapped[List[ModelMessage]] = mapped_column(JSONEncodedModelMessage, nullable=False, deferred=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())
    
    chat_history: Mapped["ChatHistoryDB"] = relationship("ChatHistoryDB", back_populates="chat")
    parent: Mapped["ChatMessageDB"] = relationship("ChatMessageDB", remote_side=[id], backref="children")
    
    __table_args__ = (
        Index("idx_chat_messages_timestamp", "timestamp"),
    )
    
class KnowledgeBaseList(TypeDecorator):
    """知识库列表类型"""
    impl = String

    def process_bind_param(self, value, dialect):
        if isinstance(value, list):
            return ','.join([str(v) for v in value])
        return None

    def process_result_value(self, value, dialect):
        if isinstance(value, str):
            return [UUID(v) for v in value.split(',')]
        return []
    
class ChatHistoryDB(Base):
    """数据库聊天历史存储模型"""
    __tablename__ = "chat_history"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    knowledge_base: Mapped[List[UUID]] = mapped_column(KnowledgeBaseList, nullable=True)
    deleted: Mapped[bool] = mapped_column(default=False, nullable=False)
    
    chat: Mapped[List["ChatMessageDB"]] = relationship("ChatMessageDB", back_populates="chat_history", order_by=ChatMessageDB.timestamp.asc())
    
    __table_args__ = (
        Index("idx_chat_histories_updated_at", "updated_at"),
    )
import uuid
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DATETIME, ForeignKey, Index, String, TIMESTAMP, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.model import Base

if TYPE_CHECKING:
    pass
    
class ChatMessageDB(Base):
    """数据库聊天消息存储模型"""
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parentId: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("chat_messages.id"), nullable=True)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    tooluse: Mapped[str] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())
    
    __table_args__ = (
        Index("idx_chat_messages_timestamp", "timestamp"),
    )
    
class ChatHistoryDB(Base):
    """数据库聊天历史存储模型"""
    __tablename__ = "chat_history"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
    
    chat: Mapped["ChatMessageDB"] = relationship("ChatMessageDB", back_populates="chat_history")
    
    __table_args__ = (
        Index("idx_chat_histories_updated_at", "updated_at"),
    )
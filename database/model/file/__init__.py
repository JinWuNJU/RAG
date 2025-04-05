from datetime import datetime
from uuid import UUID
import uuid
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, LargeBinary, String
from database.model import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.model.user import User
from sqlalchemy.dialects.postgresql import UUID as PGUUID

class FileDB(Base):
    """数据库文件存储模型"""
    __tablename__ = "files"
    # 列定义
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # 关联ORM
    user: Mapped["User"] = relationship("User")
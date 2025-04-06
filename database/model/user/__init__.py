import re
from typing import TYPE_CHECKING
from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.model import Base

if TYPE_CHECKING:
    from database.model.file import FileDB
    from database.model.knowledge_base import KnowledgeBase
    
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(100), nullable=False)
    
    files: Mapped[list["FileDB"]] = relationship("FileDB", back_populates="user")
    knowledge_bases: Mapped[list["KnowledgeBase"]] = relationship("KnowledgeBase", back_populates="uploader")
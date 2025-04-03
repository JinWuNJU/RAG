from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
import uuid


Base = declarative_base()

class KnowledgeBase(Base):
    """知识库元数据表"""
    __tablename__ = "knowledge_bases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    status = Column(String(20), default="building")  # building/completed
    chunk_size = Column(Integer, nullable=False)
    overlap_size = Column(Integer, nullable=False)
    hybrid_ratio = Column(Float, default=0.5)

class KnowledgeBaseChunk(Base):
    """知识库分块内容表"""
    __tablename__ = "knowledge_base_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_base_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id"), nullable=False)
    file_id = Column(UUID(as_uuid=True), nullable=False)
    file_name = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # 向量存储



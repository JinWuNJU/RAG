from datetime import datetime
from typing import Optional
from sqlalchemy import CheckConstraint, Boolean, Index, String, Integer, Float, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import func
import uuid
from sqlalchemy.orm import Mapped, mapped_column

from database.model import Base

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_bases'
    
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, server_default=func.gen_random_uuid())
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(20), server_default='building')
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    overlap_size: Mapped[int] = mapped_column(Integer, nullable=False)
    hybrid_ratio: Mapped[float] = mapped_column(Float, server_default='0.5')
    is_public: Mapped[bool] = mapped_column(Boolean, server_default='false', nullable=False)
    uploader_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey('users.id'), nullable=False)

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint("status IN ('building', 'completed', 'failed')", name='check_valid_status'),
        CheckConstraint('chunk_size > 0', name='check_chunk_size_positive'),
        CheckConstraint('overlap_size >= 0', name='check_overlap_size_non_negative'),
        CheckConstraint('hybrid_ratio BETWEEN 0 AND 1', name='check_hybrid_ratio_range'),
        
        # Index on status column
        Index('idx_knowledge_bases_status', 'status'),
    )

class KnowledgeBaseChunk(Base):
    __tablename__ = 'knowledge_base_chunks'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(UUID, ForeignKey('knowledge_bases.id', ondelete='CASCADE'), nullable=False)
    file_id: Mapped[uuid.UUID] = mapped_column(UUID, nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(1536), nullable=True)
    
    # Check constraint
    __table_args__ = (
        CheckConstraint('chunk_index >= 0', name='check_chunk_index_positive'),
        
        # Unique constraint
        UniqueConstraint('knowledge_base_id', 'file_id', 'chunk_index', name='idx_chunks_uniq'),
        
        # Regular indexes
        Index('idx_chunks_knowledge_base', 'knowledge_base_id'),
        Index('idx_chunks_file', 'file_id'),
        
        # PGroonga full-text search index
        Index('pgroonga_content_index', content, postgresql_using='pgroonga'),
        
        # pgvector index using IVFFlat algorithm
        Index('ivfflat_embedding_index', embedding, postgresql_using='ivfflat', 
              postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_l2_ops'}),
    )



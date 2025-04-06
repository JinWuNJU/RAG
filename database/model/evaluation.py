# database/model/evaluation.py
from sqlalchemy import Column, String, JSON, DateTime
from sqlalchemy.dialects.postgresql import UUID
from database.base import Base

class EvaluationTask(Base):
    __tablename__ = "evaluation_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String(200), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    status = Column(String(20), default="processing")
    created_at = Column(DateTime, nullable=False)
    error_message = Column(String(500))

class EvaluationRecord(Base):
    __tablename__ = "evaluation_records"

    id = Column(UUID(as_uuid=True), primary_key=True)
    task_id = Column(UUID(as_uuid=True), nullable=False)
    metric_id = Column(String(50), nullable=False)
    system_prompt = Column(String, nullable=False)
    file_id = Column(UUID(as_uuid=True), nullable=False)
    results = Column(JSON)
    created_at = Column(DateTime, nullable=False)
# database/model/evaluation.py
from sqlalchemy import Column, String, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from database.model import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship


class EvaluationTask(Base):
    __tablename__ = "evaluation_tasks"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="processing")
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
    error_message: Mapped[str | None] = mapped_column(String(500))
    is_rag_task: Mapped[bool] = mapped_column(Boolean, default=False)

    # 添加反向关系
    records = relationship("EvaluationRecord", back_populates="task")


class EvaluationRecord(Base):
    __tablename__ = "evaluation_records"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    task_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evaluation_tasks.id"), nullable=False)
    metric_id: Mapped[str] = mapped_column(String(50), nullable=False)
    system_prompt: Mapped[str] = mapped_column(String, nullable=True)
    file_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    results: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False)

    # 添加关系
    task = relationship("EvaluationTask", back_populates="records")


class CustomMetric(Base):
    __tablename__ = "custom_metrics"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    criteria: Mapped[list] = mapped_column(JSON, nullable=False)  # 评分标准列表
    instruction: Mapped[str] = mapped_column(String, nullable=False)  # 指导指令
    scale: Mapped[int] = mapped_column(JSON, nullable=False, default=10)  # 评分尺度
    type: Mapped[str] = mapped_column(String(20), nullable=False, default="custom")  # 自定义类型，可以是 custom 或 rubrics
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False)
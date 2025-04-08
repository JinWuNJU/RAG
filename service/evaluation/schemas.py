from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID

class Metric(BaseModel):
    id: str
    name: str
    description: str

class EvaluationRequest(BaseModel):
    metric_id: str
    task_name: str
    system_prompt: str
    file_id: str

class EvaluationResultItem(BaseModel):
    query: str
    answer: str
    score: float
    details: dict

class EvaluationRecordResponse(BaseModel):
    id: UUID
    created_at: str
    results: List[EvaluationResultItem]

class EvaluationRequest(BaseModel):
    name: str  # 确保字段名与前端一致
    system_prompt: str
    metric_id: str
    file_id: str

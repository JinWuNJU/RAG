from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime

class Metric(BaseModel):
    id: str
    name: str
    description: str

class EvaluationRequest(BaseModel):
    metric_id: str
    task_name: str
    system_prompt: str
    file_id: str

class EvaluationIterationRequest(BaseModel):
    task_id: str
    system_prompt: str

class EvaluationResultItem(BaseModel):
    query: str
    answer: str
    generated: Optional[str] = None
    score: float
    details: dict

class EvaluationRecordResponse(BaseModel):
    id: UUID
    task_id: str
    system_prompt: str
    created_at: Union[datetime, int, str]
    status: str
    score: Optional[float] = None
    detailed_results: Optional[Dict[str, Any]] = None

class EvaluationTaskItem(BaseModel):
    id: str
    name: str
    created_at: Union[datetime, int]
    metric_id: str
    metric_name: str
    status: str
    dataset_id: str
    iterations: int = 0

class EvaluationTasksResponse(BaseModel):
    tasks: List[EvaluationTaskItem]
    total: int

class EvaluationTaskCreateResponse(BaseModel):
    task_id: str
    record_id: str

class EvaluationIterationResponse(BaseModel):
    record_id: str

class DeleteTaskResponse(BaseModel):
    success: bool
    message: str
    created_at: str
    results: List[EvaluationResultItem]

class EvaluationRequest(BaseModel):
    name: str  # 确保字段名与前端一致
    system_prompt: str
    metric_id: str
    file_id: str

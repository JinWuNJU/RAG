from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime

class Metric(BaseModel):
    id: str
    name: str
    description: str
    type: Optional[str] = None  # 新增字段，用于标识指标类型 (prompt/rag)

# 添加自定义指标的模型
class CustomMetricDefinition(BaseModel):
    name: str  # 自定义指标名称
    description: str  # 指标描述
    criteria: List[str]  # 评分标准列表
    instruction: str  # 评分指导说明
    scale: int = 10  # 评分尺度，默认10分制
    type: str = "custom"  # 指标类型，custom或rubrics

# 创建自定义指标请求
class CustomMetricRequest(BaseModel):
    metric_definition: CustomMetricDefinition

# 添加自定义指标的评估请求
class CustomMetricEvaluationRequest(BaseModel):
    task_id: str
    system_prompt: str
    custom_metric_id: str  # 自定义指标ID

class EvaluationRequest(BaseModel):
    task_name: str
    file_id: str
    metric_id: str
    system_prompt: str

class EvaluationIterationRequest(BaseModel):
    task_id: str
    system_prompt: str

# 新增RAG评估相关模型
class RAGEvaluationRequest(BaseModel):
    task_name: str
    file_id: str
    metric_ids: List[str]  # 可以选择多个评估指标

class RAGIterationRequest(BaseModel):
    task_id: str
    file_id: str  # RAG评估需要上传新的文件而不是修改prompt

class RAGSampleItem(BaseModel):
    query: str
    answer: str
    retrieved_contexts: List[str]
    ground_truth: Optional[str] = None

class EvaluationRecordResponse(BaseModel):
    id: str
    task_id: str
    system_prompt: Optional[str] = None
    created_at: int  # 时间戳
    status: str
    score: Optional[float] = None
    detailed_results: Optional[Dict[str, Any]] = None
    
class EvaluationTasksResponse(BaseModel):
    tasks: List[Any]
    total: int

class EvaluationTaskItem(BaseModel):
    id: str
    name: str
    created_at: int  # 时间戳
    metric_id: str = ""
    metric_name: str = ""
    status: str
    dataset_id: str = ""
    iterations: int = 0
    is_rag_task: bool = False  # 新增标识是否为RAG评估任务的字段

class EvaluationTaskCreateResponse(BaseModel):
    task_id: str
    record_id: str

class EvaluationIterationResponse(BaseModel):
    record_id: str

class DeleteTaskResponse(BaseModel):
    success: bool
    message: str

# 自定义指标创建响应
class CustomMetricCreateResponse(BaseModel):
    metric_id: str
    name: str

# 新增RAG评估任务项模型，专门用于RAG评估API返回
class RAGEvaluationTaskItem(BaseModel):
    id: str
    name: str
    created_at: int  # 时间戳
    metric_id: str = ""
    metric_ids: Optional[List[str]] = None  # RAG评估可能使用多个指标
    metric_name: str = ""
    status: str
    dataset_id: str = ""  # 保持向后兼容
    file_id: str = ""     # 新字段名称
    iterations: int = 0
    is_rag_task: bool = True  # 始终为True

# 新增RAG评估任务列表响应模型
class RAGEvaluationTasksResponse(BaseModel):
    tasks: List[RAGEvaluationTaskItem]
    total: int
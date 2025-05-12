from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Metric(BaseModel):
    id: str  # 指标的唯一标识符
    name: str  # 指标的显示名称
    description: str  # 详细描述指标的评估内容和目的
    type: Optional[str] = None  # 指标类型：'prompt'用于提示词评估, 'rag'用于RAG系统评估, 'custom'用于自定义评估

# 自定义评估指标模型
class CustomMetricDefinition(BaseModel):
    name: str  # 自定义指标名称，如"答案清晰度"、"专业准确性"等
    description: str  # 详细描述此指标的评估目标和应用场景
    criteria: List[str]  # 评分标准列表，如["信息准确性","答案完整性","表达清晰度"]
    instruction: str  # 给评估模型的详细指导说明，用于指导如何评价生成内容
    scale: int = 10  # 评分量表，默认为10分制
    type: str = "custom"  # 指标类型：'custom'为自定义指标，'rubrics'为评分标准

# 创建自定义指标的请求模型
class CustomMetricRequest(BaseModel):
    metric_definition: CustomMetricDefinition  # 包含自定义指标的完整定义

# 使用自定义指标进行评估的请求模型
class CustomMetricEvaluationRequest(BaseModel):
    task_id: str  # 要评估的任务ID
    system_prompt: str  # 用于生成回答的系统提示词
    custom_metric_id: str  # 要使用的自定义指标ID

# Prompt评估请求模型
class EvaluationRequest(BaseModel):
    task_name: str  # 评估任务名称，用于标识和区分不同的评估任务
    file_id: str  # 包含评估数据的文件ID
    metric_id: str  # 要使用的评估指标ID
    system_prompt: str  # 用于生成回答的系统提示词

# Prompt评估迭代请求模型（用于优化已有任务）
class EvaluationIterationRequest(BaseModel):
    task_id: str  # 要迭代的评估任务ID
    system_prompt: str  # 新的系统提示词，用于改进生成结果

# RAG评估请求模型
class RAGEvaluationRequest(BaseModel):
    task_name: str  # RAG评估任务名称
    file_id: str  # 包含评估数据的文件ID
    metric_ids: List[str]  # 要使用的评估指标ID列表，可选多个指标同时评估

# RAG评估迭代请求模型
class RAGIterationRequest(BaseModel):
    task_id: str  # 要迭代的RAG评估任务ID
    file_id: str  # 新的评估数据文件ID（RAG评估需要提供新文件而非修改提示词）

# RAG评估样本项模型
class RAGSampleItem(BaseModel):
    query: str  # 用户查询问题
    answer: str  # 系统生成的回答
    retrieved_contexts: List[str]  # 检索到的上下文片段列表
    ground_truth: Optional[str] = None  # 可选的标准答案

# 评估记录响应模型
class EvaluationRecordResponse(BaseModel):
    id: str  # 评估记录的唯一ID
    task_id: str  # 对应评估任务的ID
    system_prompt: Optional[str] = None  # 用于生成回答的系统提示词
    created_at: int  # 创建时间戳（毫秒）
    status: str  # 评估状态：'processing'、'completed'或'failed'
    score: Optional[float] = None  # 总体评分（0-1之间）
    detailed_results: Optional[Dict[str, Any]] = None  # 详细评估结果，包含各指标分数
    
# 评估任务列表响应模型
class EvaluationTasksResponse(BaseModel):
    tasks: List[Any]  # 评估任务列表
    total: int  # 总任务数

# 评估任务详情模型
class EvaluationTaskItem(BaseModel):
    id: str  # 任务唯一ID
    name: str  # 任务名称
    created_at: int  # 创建时间戳（毫秒）
    metric_id: str = ""  # 使用的评估指标ID
    metric_name: str = ""  # 评估指标名称
    status: str  # 任务状态：'processing'、'completed'或'failed'
    dataset_id: str = ""  # 数据集ID
    iterations: int = 0  # 迭代次数，表示任务已优化的次数
    is_rag_task: bool = False  # 是否为RAG评估任务

# 创建评估任务响应模型
class EvaluationTaskCreateResponse(BaseModel):
    task_id: str  # 新创建的评估任务ID
    record_id: str  # 初始评估记录ID

# 评估迭代响应模型
class EvaluationIterationResponse(BaseModel):
    record_id: str  # 新创建的评估记录ID

# 删除任务响应模型
class DeleteTaskResponse(BaseModel):
    success: bool  # 操作是否成功
    message: str  # 成功或失败的消息

# 自定义指标创建响应模型
class CustomMetricCreateResponse(BaseModel):
    metric_id: str  # 新创建的自定义指标ID
    name: str  # 指标名称

# RAG评估任务详情模型
class RAGEvaluationTaskItem(BaseModel):
    id: str  # 任务唯一ID
    name: str  # 任务名称
    created_at: int  # 创建时间戳（毫秒）
    metric_id: str = ""  # 兼容旧版API的单一指标ID
    metric_ids: Optional[List[str]] = None  # RAG评估使用的多个指标ID列表
    metric_name: str = ""  # 主要指标名称（兼容旧版API）
    status: str  # 任务状态
    dataset_id: str = ""  # 旧版API兼容字段
    file_id: str = ""  # 数据文件ID
    iterations: int = 0  # 迭代次数
    is_rag_task: bool = True  # 始终为True，表示这是RAG评估任务

# RAG评估任务列表响应模型
class RAGEvaluationTasksResponse(BaseModel):
    tasks: List[RAGEvaluationTaskItem]  # RAG评估任务列表
    total: int  # 总任务数
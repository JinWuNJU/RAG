# RAG和Prompt评估系统

这个模块提供了一个全面的评估框架，用于评估检索增强生成(RAG)系统和提示词(Prompt)优化的效果。系统支持多种内置评估指标，以及创建和使用自定义评估指标。

## 功能概述

### Prompt评估功能
- 评估不同系统提示词(System Prompts)生成回答的质量
- 支持多种评估指标，如BLEU分数、答案相关性等
- 可进行多次迭代优化提示词
- 提供详细的评估报告和分数

### RAG评估功能
- 评估检索增强生成系统的效果
- 支持评估检索上下文的相关性和精确度
- 评估生成答案对上下文的忠实度
- 可同时使用多个评估指标

### 自定义评估功能
- 创建自定义评估指标和标准
- 自定义评分量表和评估标准
- 使用特定领域知识进行专业化评估

## 评估指标说明

### Prompt评估指标
1. **答案相关性 (answer_relevancy)**
   - 评估生成的答案与问题的相关程度
   - 检查答案是否直接回应了问题
   - 检查是否包含无关内容

2. **Prompt调优评分 (prompt_scs)**
   - 由LLM评判生成答案与参考答案的相似度
   - 考虑准确性、完整性、相关性和清晰度
   - 提供0-10分的综合评分

3. **BLEU分数 (bleu)**
   - 计算生成答案与参考答案的文本相似度
   - 基于N-gram匹配计算分数
   - 适合评估翻译质量和文本生成准确性

### RAG评估指标
1. **忠实度 (faithfulness)**
   - 评估答案是否忠实于被召回的上下文
   - 检查答案中的信息是否均来自上下文
   - 检测是否存在"幻觉"(hallucination)

2. **上下文相关性 (context_relevancy)**
   - 评估检索的上下文与查询问题的相关程度
   - 判断上下文是否包含回答问题所需的关键信息
   - 评估召回的内容与用户意图的匹配度

3. **上下文精确度 (context_precision)**
   - 评估召回的上下文中有用信息的比例
   - 检查上下文是否简洁且集中
   - 评估冗余或无关信息的多少

### 自定义评估指标
- 可根据特定需求创建自定义评估指标
- 支持多个评分标准
- 可定制评分指导说明和评分量表

## 使用指南

### 创建Prompt评估任务
1. 准备包含问题和参考答案的数据集文件
2. 选择合适的评估指标
3. 提供系统提示词
4. 创建评估任务并等待结果

### 创建RAG评估任务
1. 准备包含查询、生成答案和检索上下文的数据集文件
2. 选择一个或多个RAG评估指标
3. 创建评估任务并等待结果

### 创建和使用自定义评估指标
1. 定义评估标准和指导说明
2. 设置评分量表(默认为10分制)
3. 创建自定义指标
4. 在评估任务中选择该自定义指标

## API参考

### 评估指标相关
- `GET /evaluation/metrics`: 获取所有可用评估指标
- `POST /evaluation/custom_metrics`: 创建新的自定义评估指标
- `GET /evaluation/custom_metrics`: 获取所有自定义评估指标

### Prompt评估相关
- `POST /evaluation/tasks`: 创建新的Prompt评估任务
- `POST /evaluation/iterations`: 创建Prompt评估迭代(优化已有任务)
- `POST /evaluation/custom_iterations`: 使用自定义指标进行评估迭代

### RAG评估相关
- `POST /evaluation/rag/tasks`: 创建新的RAG评估任务
- `POST /evaluation/rag/iterations`: 创建RAG评估迭代(使用新数据)

### 任务和记录管理
- `GET /evaluation/tasks`: 获取评估任务列表
- `GET /evaluation/tasks/{task_id}`: 获取指定评估任务的详情
- `GET /evaluation/tasks/{task_id}/records`: 获取任务的所有评估记录
- `GET /evaluation/records/{record_id}`: 获取指定评估记录的详情
- `DELETE /evaluation/tasks/{task_id}`: 删除评估任务

## 数据模型

### 评估任务 (EvaluationTask)
- 包含任务名称、状态、创建时间等基本信息
- 关联多个评估记录
- 区分普通Prompt评估任务和RAG评估任务

### 评估记录 (EvaluationRecord)
- 记录单次评估的详细结果
- 包含使用的指标、系统提示词和评分
- 关联到特定评估任务

### 自定义指标 (CustomMetric)
- 定义评估标准和方法
- 包含名称、描述、评分标准和指导说明
- 可用于Prompt评估和RAG评估

## 实现细节

本评估系统使用LLM作为评估器，结合统计方法进行评分。主要包括:

1. **基于LLM的评估**:
   - 使用指定的评估LLM模型进行主观评价
   - 提供详细的评估指导以确保一致性
   - 适用于需要理解和判断的评估指标

2. **统计方法评估**:
   - 使用BLEU等算法计算文本相似度
   - 适用于客观评估生成文本的质量
   - 提供稳定和可复现的评分

3. **混合评估**:
   - 结合LLM评估和统计方法
   - 提供更全面的质量评价
   - 自动容错和备选评分机制

## 部署和配置

系统需要以下环境变量进行配置:

```
EVAL_LLM_API_KEY=您的评估LLM API密钥
EVAL_LLM_API_ENDPOINT=评估LLM的API端点，默认为"https://open.bigmodel.cn/api/paas/v4/"
EVAL_LLM_MODEL_ID=评估使用的模型ID，默认为"glm-4-flash-250414"
```

## 使用示例

### 创建Prompt评估任务
```json
POST /evaluation/tasks
{
  "task_name": "客户服务回复质量评估",
  "file_id": "f3a2b1c0-8d4e-4f7a-9c2b-1e3d4f5a6b7c",
  "metric_id": "prompt_scs",
  "system_prompt": "你是一名专业的客户服务代表，请针对客户问题提供友好、准确的回答。"
}
```

### 创建RAG评估任务
```json
POST /evaluation/rag/tasks
{
  "task_name": "产品知识库RAG评估",
  "file_id": "d1c2b3a4-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
  "metric_ids": ["faithfulness", "context_relevancy", "context_precision"]
}
```

### 创建自定义评估指标
```json
POST /evaluation/custom_metrics
{
  "metric_definition": {
    "name": "专业术语准确性",
    "description": "评估回答中专业术语使用的准确性和适当性",
    "criteria": [
      "术语定义正确性",
      "术语使用合适性",
      "专业概念解释清晰度"
    ],
    "instruction": "评估回答中专业术语的使用是否准确、适当，以及专业概念是否解释清晰",
    "scale": 10,
    "type": "custom"
  }
}
``` 
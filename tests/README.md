# RAG后端单元测试

本目录包含RAG后端项目的单元测试，特别关注评估功能的测试。

## 测试文件结构

- `conftest.py` - 包含共享的测试fixture，设置测试环境、模拟JWT认证和数据库会话
- `test_evaluation.py` - 测试Prompt评估功能
- `test_rag_evaluation.py` - 测试RAG评估功能
- `test_custom_metrics.py` - 测试自定义评估指标功能
- `test_rag_simple.py` - 测试RAG简单功能和辅助函数
- `test_end_to_end.py` - 测试完整评估流程，使用实际用户ID
- `test_prompt_evaluator.py` - 专门测试SimplePromptEvaluator类
- `mock_server.py` - 提供模拟API服务器用于测试

## 测试设计

测试使用了unittest.mock模块模拟数据库连接、JWT认证和服务调用，避免实际调用外部API。测试覆盖了评估功能的完整流程，包括：

1. 创建评估任务
2. 获取评估结果
3. 创建评估迭代
4. 删除评估任务
5. 自定义评估指标创建和使用


## 运行测试

可以使用以下命令运行测试：

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_evaluation.py

# 运行特定测试类
pytest tests/test_evaluation.py::TestEvaluationService

# 运行特定测试方法
pytest tests/test_evaluation.py::TestEvaluationService::test_evaluate
```

## 测试依赖

测试依赖于以下库：

- pytest
- pytest-asyncio
- unittest.mock

## 测试覆盖范围

当前测试覆盖了以下功能：

1. **Prompt评估功能**
   - 评估服务初始化
   - 评估方法
   - 任务管理（创建、获取、删除）
   - 简单提示词评估器

2. **RAG评估功能**
   - RAG评估方法
   - 忠实度评估
   - 上下文相关性评估
   - 上下文精确度评估
   - RAG任务管理

3. **自定义评估指标功能**
   - 创建自定义评估指标
   - 使用自定义指标评估
   - 自定义指标管理

4. **辅助功能**
   - BLEU评分计算
   - 简单相似度计算
   - 指标分类获取

5. **端到端流程**
   - 标准评估完整流程（创建→获取→删除）
   - RAG评估完整流程（创建→迭代→获取）
   - 自定义指标完整流程（创建→获取→评估）
   - 使用实际用户ID进行测试

## 常见问题解决

1. **导入错误**：如果遇到导入错误，确保已将项目根目录添加到系统路径中
2. **数据库连接问题**：测试使用模拟数据库会话，不需要实际连接数据库
3. **异步测试问题**：确保使用了`pytest.mark.asyncio`装饰器和`pytest-asyncio`插件 
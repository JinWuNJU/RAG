import os
import sys
import uuid
import json
import pytest
import random
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.evaluation.service import EvaluationService
from database.model.evaluation import EvaluationTask, EvaluationRecord


# 为所有测试设置固定的随机数种子
@pytest.fixture(autouse=True)
def set_random_seed():
    """自动使用固定的随机数种子以确保测试结果稳定"""
    random.seed(42)
    np.random.seed(42)


class TestRAGEvaluation:
    """
    测试RAG评价功能
    """
    
    @pytest.mark.asyncio
    async def test_evaluate_rag(self, mock_db):
        """
        测试RAG评价方法
        """
        service = EvaluationService(mock_db)
        
        # 使用固定返回值而不是随机值
        # 仅模拟外部依赖，保留真实代码逻辑
        with patch.object(service, '_evaluate_faithfulness', return_value=[0.8, 0.8, 0.8]):
            with patch.object(service, '_evaluate_context_relevancy', return_value=[0.7, 0.7, 0.7]):
                with patch.object(service, '_evaluate_context_precision', return_value=[0.9, 0.9, 0.9]):
                    # 测试数据 - 使用正确的键名 retrieved_contexts
                    data = [
                        {
                            "query": "什么是人工智能？",
                            "answer": "人工智能是研究如何使计算机能够像人一样思考和学习的科学。",
                            "retrieved_contexts": ["人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。"]
                        }
                    ]
                    
                    # 调用评估方法
                    result = await service.evaluate_rag(
                        data,
                        ["faithfulness", "context_relevancy", "context_precision"]
                    )
                    
                    # 验证结果结构 - 修正检查方式以适应实际返回类型
                    assert isinstance(result, object)
                    assert hasattr(result, 'scores')
                    scores = result.scores
                    
                    # 验证指标分数
                    assert "faithfulness" in scores
                    assert "context_relevancy" in scores
                    assert "context_precision" in scores
                    
                    # 验证分数值 - 使用模拟返回值对比
                    assert scores["faithfulness"] == [0.8, 0.8, 0.8]
                    assert scores["context_relevancy"] == [0.7, 0.7, 0.7]
                    assert scores["context_precision"] == [0.9, 0.9, 0.9]
    
    @pytest.mark.asyncio
    async def test_evaluate_faithfulness(self, mock_db):
        """
        测试忠实度评价
        """
        service = EvaluationService(mock_db)
        
        # 测试数据
        queries = ["什么是人工智能？"]
        answers = ["人工智能是研究如何使计算机能够像人一样思考和学习的科学。"]
        contexts = [["人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。"]]
        
        # 完全模拟调用链，避免调用真实API
        with patch('service.evaluation.service.ChatPromptTemplate') as mock_template:
            with patch.object(service, 'llm') as mock_llm:
                # 设置固定返回值
                mock_chain = MagicMock()
                mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="8.5"))
                mock_template.from_messages.return_value = mock_template
                mock_template.__or__.return_value = mock_chain
                
                # 调用真实方法
                result = await service._evaluate_faithfulness(queries, answers, contexts)
                
                # 验证结果，用精确值断言
                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0] == 0.85
    
    @pytest.mark.asyncio
    async def test_evaluate_context_relevancy(self, mock_db):
        """
        测试上下文相关性评价
        """
        service = EvaluationService(mock_db)
        
        # 测试数据
        queries = ["什么是人工智能？"]
        contexts = [["人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。"]]
        
        # 完全模拟调用链，避免调用真实API
        with patch('service.evaluation.service.ChatPromptTemplate') as mock_template:
            with patch.object(service, 'llm') as mock_llm:
                # 设置固定返回值
                mock_chain = MagicMock()
                mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="7.6"))
                mock_template.from_messages.return_value = mock_template
                mock_template.__or__.return_value = mock_chain
                
                # 调用真实方法
                result = await service._evaluate_context_relevancy(queries, contexts)
                
                # 验证结果，用精确值断言
                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0] == 0.76
    
    @pytest.mark.asyncio
    async def test_evaluate_context_precision(self, mock_db):
        """
        测试上下文精确度评价
        """
        service = EvaluationService(mock_db)
        
        # 测试数据
        queries = ["什么是人工智能？"]
        answers = ["人工智能是研究如何使计算机能够像人一样思考和学习的科学。"]
        contexts = [["人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。"]]
        
        # 完全模拟调用链，避免调用真实API
        with patch('service.evaluation.service.ChatPromptTemplate') as mock_template:
            with patch.object(service, 'llm') as mock_llm:
                # 设置固定返回值
                mock_chain = MagicMock()
                mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="9.2"))
                mock_template.from_messages.return_value = mock_template
                mock_template.__or__.return_value = mock_chain
                
                # 调用真实方法
                result = await service._evaluate_context_precision(queries, answers, contexts)
                
                # 验证结果，用精确值断言
                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0] == 0.92
    
    @pytest.mark.asyncio
    async def test_rag_task_records(self, mock_db):
        """
        测试获取RAG任务记录
        """
        service = EvaluationService(mock_db)
        
        # 设置模拟数据
        task_id = uuid.uuid4()
        user_id = uuid.uuid4()
        now = datetime.now()
    
        # 创建模拟RAG任务
        mock_task = MagicMock(spec=EvaluationTask)
        mock_task.id = task_id
        mock_task.name = "测试RAG任务"
        mock_task.user_id = user_id
        mock_task.status = "completed"
        mock_task.created_at = now
        mock_task.is_rag_task = True
        
        # 创建模拟评估记录 - 使用固定结构和值
        mock_records = []
        metric_ids = ["faithfulness", "context_relevancy", "context_precision"]
        
        for idx, metric_id in enumerate(metric_ids):
            record = MagicMock(spec=EvaluationRecord)
            record.id = uuid.uuid4()
            record.task_id = task_id
            record.metric_id = metric_id
            record.file_id = uuid.uuid4()
            # 使用固定的模拟结果
            fixed_scores = {
                "faithfulness": 0.85,
                "context_relevancy": 0.78, 
                "context_precision": 0.92
            }
            # 确保结构与真实代码处理匹配
            record.results = {
                "scores": {
                    metric_id: fixed_scores[metric_id]  # 使用固定值而非随机数
                }
            }
            record.created_at = now
            record.metric_name = f"{metric_id}_名称"
            mock_records.append(record)
        
        # 创建DB查询模拟
        def custom_query(model_class):
            mock_query = MagicMock()
            # 任务查询
            if model_class == EvaluationTask:
                task_filter = MagicMock()
                task_filter.first.return_value = mock_task
                mock_query.filter.return_value = task_filter
            # 记录查询
            elif model_class == EvaluationRecord:
                record_filter = MagicMock()
                record_order = MagicMock()
                record_order.all.return_value = mock_records
                record_filter.order_by.return_value = record_order
                mock_query.filter.return_value = record_filter
            return mock_query
        
        # 替换mock_db.query方法
        mock_db.query = custom_query
        
        # 用patch替换get_metric_name函数以返回固定值
        with patch.object(service, 'metrics', {
            "faithfulness": {"name": "忠实度"},
            "context_relevancy": {"name": "上下文相关性"},
            "context_precision": {"name": "上下文精确度"}
        }):
            records = service.get_task_records(task_id, user_id)
            
            # 验证结果
            assert len(records) == 3
            
            # 验证返回的记录结构和值
            for idx, record in enumerate(records):
                assert isinstance(record, dict)
                
                # 检查关键字段存在
                assert "id" in record
                assert "created_at" in record
                assert "score" in record
                
                # 验证分数值是固定的
                expected_scores = [0.85, 0.78, 0.92]
                assert record["score"] == expected_scores[idx]
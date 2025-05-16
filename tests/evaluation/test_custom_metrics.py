import os
import sys
import uuid
import pytest
import random
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.evaluation.service import EvaluationService
from database.model.evaluation import CustomMetric
from service.evaluation.schemas import CustomMetricDefinition


# 为所有测试设置固定的随机数种子
@pytest.fixture(autouse=True)
def set_random_seed():
    """自动使用固定的随机数种子以确保测试结果稳定"""
    random.seed(42)
    np.random.seed(42)


class TestCustomMetrics:
    """
    测试自定义评估指标功能
    """
    
    @pytest.mark.asyncio
    async def test_create_custom_metric(self, mock_db, test_user_id):
        """
        测试创建自定义评估指标
        """
        service = EvaluationService(mock_db)
        
        # 使用固定的UUID而不是随机生成
        fixed_metric_uuid = uuid.UUID('12345678-1234-5678-1234-567812345678')
        now = datetime(2023, 5, 15, 10, 30, 0)  # 使用固定的时间
        
        # 模拟uuid和时间，但保留真实创建逻辑
        with patch('uuid.uuid4', return_value=fixed_metric_uuid):
            with patch('service.evaluation.service.get_beijing_time', return_value=now):
                # 定义测试指标
                metric_definition = CustomMetricDefinition(
                    name="测试指标",
                    description="测试指标描述",
                    criteria=["准确性", "相关性"],
                    instruction="评估指令",
                    scale=10,
                    type="custom"
                )
                
                # 调用真实方法
                result = await service.create_custom_metric(
                    user_id=test_user_id, 
                    metric_definition=metric_definition
                )
                
                # 验证结果格式和内容
                assert "metric_id" in result
                assert "name" in result
                assert result["metric_id"] == f"custom_{str(fixed_metric_uuid)}"
                assert result["name"] == "测试指标"
                
                # 验证数据库操作 - 检查添加到数据库的CustomMetric对象
                mock_db.add.assert_called_once()
                added_metric = mock_db.add.call_args[0][0]
                assert isinstance(added_metric, CustomMetric)
                assert added_metric.id == fixed_metric_uuid
                assert added_metric.user_id == test_user_id
                assert added_metric.name == "测试指标"
                assert added_metric.description == "测试指标描述"
                assert added_metric.criteria == ["准确性", "相关性"]
                assert added_metric.instruction == "评估指令"
                assert added_metric.scale == 10
                assert added_metric.type == "custom"
    
    @pytest.mark.asyncio
    async def test_evaluate_with_custom_metric(self, mock_db):
        """
        测试使用自定义指标评估
        """
        service = EvaluationService(mock_db)
        
        # 测试数据
        questions = ["什么是人工智能？"]
        answers = ["人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。"]
        generated_responses = ["人工智能是研究如何使计算机能够像人一样思考和学习的科学。"]
        
        # 自定义指标信息
        metric_info = {
            "id": "custom_12345678-1234-5678-1234-567812345678",
            "name": "测试指标",
            "description": "测试指标描述",
            "criteria": ["准确性", "相关性"],
            "instruction": "评估指令",
            "scale": 10,
            "type": "custom"
        }
        
        # 使用ChatPromptTemplate模拟实际的函数调用
        with patch('service.evaluation.service.ChatPromptTemplate') as mock_template:
            with patch.object(service, 'llm') as mock_llm:
                # 设置固定返回值
                mock_chain = MagicMock()
                mock_chain.ainvoke = AsyncMock(return_value=MagicMock(content="8.2"))
                mock_template.from_messages.return_value = mock_template
                mock_template.__or__.return_value = mock_chain
                
                # 调用真实方法
                result = await service._evaluate_with_custom_metric(
                    questions, answers, generated_responses, metric_info
                )
                
                # 验证结果
                assert isinstance(result, list)
                assert len(result) == 1
                assert result[0] == 0.82  # 精确断言值
                
                # 验证模板参数
                mock_template.from_messages.assert_called_once()
                mock_chain.ainvoke.assert_called_once()
                # 检查调用参数包含了正确的问题和回答
                call_args = mock_chain.ainvoke.call_args[0][0]
                assert "question" in call_args
                assert "reference" in call_args
                assert "generated" in call_args
    
    @pytest.mark.asyncio
    async def test_save_custom_metric(self, mock_db):
        """
        测试保存自定义指标到metrics字典
        """
        service = EvaluationService(mock_db)
        
        # 创建固定的UUID
        metric_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
        user_id = uuid.UUID('87654321-8765-4321-8765-432187654321')
        fixed_date = datetime(2023, 5, 15, 10, 30, 0)
        
        # 创建自定义指标
        custom_metric = CustomMetric(
            id=metric_id,
            user_id=user_id,
            name="测试自定义指标",
            description="测试指标描述",
            criteria=["准确性", "相关性"],
            instruction="评估指令",
            scale=10,
            type="custom",
            created_at=fixed_date
        )
        
        # 设置模拟查询返回自定义指标
        mock_db.query.return_value.all.return_value = [custom_metric]
        
        # 调用加载自定义指标方法
        service._load_custom_metrics()
        
        # 验证metrics字典中包含自定义指标
        custom_key = f"custom_{str(metric_id)}"
        assert custom_key in service.metrics
        loaded_metric = service.metrics[custom_key]
        assert loaded_metric["name"] == "测试自定义指标"
        assert loaded_metric["description"] == "测试指标描述"
        assert loaded_metric["criteria"] == ["准确性", "相关性"]
        assert loaded_metric["instruction"] == "评估指令"
        assert loaded_metric["scale"] == 10
        assert loaded_metric["type"] == "custom"
    
    @pytest.mark.asyncio
    async def test_custom_metric_in_get_metrics_by_type(self, mock_db):
        """
        测试自定义指标在get_metrics_by_type中的处理
        """
        service = EvaluationService(mock_db)
        
        # 使用固定的UUID
        metric_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
        custom_key = f"custom_{str(metric_id)}"
        
        # 创建固定的指标字典
        fixed_metrics = {
            "prompt_scs": {
                "name": "Prompt调优评分",
                "description": "使用AI评判模型综合评价",
                "type": "prompt"
            },
            "faithfulness": {
                "name": "忠实度",
                "description": "评估生成答案是否忠实于检索到的上下文",
                "type": "rag"
            },
            custom_key: {
                "name": "测试自定义指标",
                "description": "测试指标描述",
                "criteria": ["准确性", "相关性"],
                "instruction": "评估指令",
                "scale": 10,
                "type": "custom"
            }
        }
        
        # 通过patch模拟metrics字典
        with patch.object(service, 'metrics', fixed_metrics):
            # 调用方法获取自定义类型指标
            custom_metrics = service.get_metrics_by_type("custom")
            
            # 验证结果
            assert len(custom_metrics) == 1
            assert custom_key in custom_metrics
            assert custom_metrics[custom_key]["name"] == "测试自定义指标"
            assert custom_metrics[custom_key]["type"] == "custom"
            
            # 验证其他类型的指标过滤结果
            prompt_metrics = service.get_metrics_by_type("prompt")
            assert len(prompt_metrics) == 1
            assert "prompt_scs" in prompt_metrics
            
            rag_metrics = service.get_metrics_by_type("rag")
            assert len(rag_metrics) == 1
            assert "faithfulness" in rag_metrics
import os
import sys
import uuid
import random
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.evaluation.service import EvaluationService, calculate_bleu_score, calculate_simple_similarity


# 为所有测试设置固定的随机数种子
@pytest.fixture(autouse=True)
def set_random_seed():
    """自动使用固定的随机数种子以确保测试结果稳定"""
    random.seed(42)
    np.random.seed(42)


class TestRAGSimpleFunctions:
    """
    测试RAG相关的简单函数
    """
    
    def test_calculate_bleu_score(self):
        """
        测试BLEU评分计算
        """
        # 准备测试数据
        reference = "人工智能是一门研究如何使计算机能够像人一样思考和学习的学科。"
        candidate = "人工智能研究如何让计算机像人类一样思考。"
        
        # 调用真实函数
        score = calculate_bleu_score(reference, candidate)
        
        # 验证结果
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_calculate_simple_similarity(self):
        """
        测试简单相似度计算
        """
        # 准备测试数据
        reference = "人工智能是一门研究如何使计算机能够像人一样思考和学习的学科。"
        candidate = "人工智能研究如何让计算机像人类一样思考。"
        
        # 调用真实函数
        score = calculate_simple_similarity(reference, candidate)
        
        # 验证结果
        assert isinstance(score, float)
        assert 0 <= score <= 1


class TestRAGSimpleService:
    """
    测试RAG相关的Service简单方法
    """
    
    def test_get_metrics_by_type(self, mock_db):
        """
        测试按类型获取评估指标
        """
        service = EvaluationService(mock_db)
        
        # 设置固定的指标字典
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
            "custom_metric": {
                "name": "自定义指标",
                "description": "自定义评估指标",
                "type": "custom"
            }
        }
        
        # 使用patch替换真实指标字典
        with patch.object(service, 'metrics', fixed_metrics):
            # 获取不同类型的指标
            prompt_metrics = service.get_metrics_by_type("prompt")
            rag_metrics = service.get_metrics_by_type("rag")
            custom_metrics = service.get_metrics_by_type("custom")
            
            # 验证结果
            assert len(prompt_metrics) == 1
            assert "prompt_scs" in prompt_metrics
            
            assert len(rag_metrics) == 1
            assert "faithfulness" in rag_metrics
            
            assert len(custom_metrics) == 1
            assert "custom_metric" in custom_metrics
    
    def test_evaluate_with_bleu(self, mock_db):
        """
        测试使用BLEU评估
        """
        service = EvaluationService(mock_db)
        
        # 准备测试数据
        references = ["人工智能是一门研究如何使计算机能够像人一样思考和学习的学科。"]
        generated_responses = ["人工智能研究如何让计算机像人类一样思考。"]
        
        # 调用真实方法
        scores = service._evaluate_with_bleu(references, generated_responses)
        
        # 验证结果
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert 0 <= scores[0] <= 1
    
    @pytest.mark.asyncio
    async def test_mock_evaluation_result(self, mock_db):
        """
        测试获取模拟评估结果
        """
        service = EvaluationService(mock_db)
    
        # 测试获取模拟结果
        questions = ["问题1", "问题2", "问题3"]
        metric_id = "prompt_scs"
    
        # 固定随机种子确保一致性
        random.seed(42)
        np.random.seed(42)
        
        # 调用方法获取模拟结果
        result = service._get_mock_evaluation_result(questions, metric_id)
    
        # 验证结果
        assert hasattr(result, 'scores')
        assert metric_id in result.scores
        
        # 修改验证逻辑，同时支持列表和浮点数
        scores = result.scores[metric_id]
        if isinstance(scores, list):
            # 如果返回列表，验证长度
            assert len(scores) == len(questions)
            # 验证分数范围
            for score in scores:
                assert 0 <= score <= 1
        else:
            # 如果返回单一浮点数，验证范围
            assert isinstance(scores, float)
            assert 0 <= scores <= 1
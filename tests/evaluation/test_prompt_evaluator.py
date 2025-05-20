import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.evaluation.service import SimplePromptEvaluator


class TestSimplePromptEvaluator:
    """
    测试SimplePromptEvaluator类
    """
    
    @pytest.mark.asyncio
    async def test_init(self):
        """
        测试初始化评价器
        """
        # 模拟ChatOpenAI客户端
        mock_client = MagicMock()
        
        with patch('service.evaluation.service.ChatOpenAI', return_value=mock_client):
            with patch('service.evaluation.service.ChatPromptTemplate.from_messages', return_value=MagicMock()):
                evaluator = SimplePromptEvaluator("fake_api_key", "https://fake-api.com", "test-model")
                
                # 验证客户端创建逻辑
                assert evaluator.llm == mock_client
    
    @pytest.mark.asyncio
    async def test_evaluate_answer(self):
        """
        测试评价答案 - LLM返回合法分数
        """
        with patch('service.evaluation.service.ChatOpenAI', return_value=MagicMock()):
            with patch('service.evaluation.service.ChatPromptTemplate.from_messages', return_value=MagicMock()):
                evaluator = SimplePromptEvaluator("fake_api_key", "https://fake-api.com", "test-model")
                # mock chain.ainvoke 返回合法分数
                mock_response = MagicMock()
                mock_response.content = "8.5"
                evaluator.chain = MagicMock()
                evaluator.chain.ainvoke = AsyncMock(return_value=mock_response)

                score = await evaluator.evaluate_answer("问题", "参考", "生成")
                assert score == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_answer_with_invalid_response(self):
        """
        测试处理无效响应 - LLM返回无法解析的内容
        """
        with patch('service.evaluation.service.ChatOpenAI', return_value=MagicMock()):
            with patch('service.evaluation.service.ChatPromptTemplate.from_messages', return_value=MagicMock()):
                evaluator = SimplePromptEvaluator("fake_api_key", "https://fake-api.com", "test-model")
                # mock chain.ainvoke 返回非法分数
                mock_response = MagicMock()
                mock_response.content = "invalid"
                evaluator.chain = MagicMock()
                evaluator.chain.ainvoke = AsyncMock(return_value=mock_response)

                score = await evaluator.evaluate_answer("问题", "参考", "生成")
                assert score == 0.5
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_with_numeric_response(self):
        """
        测试处理包含数字的响应
        """
        with patch('service.evaluation.service.ChatOpenAI', return_value=MagicMock()):
            with patch('service.evaluation.service.ChatPromptTemplate.from_messages', return_value=MagicMock()):
                # 创建评估器实例
                evaluator = SimplePromptEvaluator("fake_api_key", "https://fake-api.com", "test-model")
                
                # 模拟链的响应
                mock_response = MagicMock()
                mock_response.content = "8.5"
                
                # 模拟链的调用
                with patch.object(evaluator, 'chain', MagicMock()) as mock_chain:
                    mock_chain.ainvoke = AsyncMock(return_value=mock_response)
                    
                    # 调用评估方法
                    score = await evaluator.evaluate_answer("测试问题", "测试参考", "测试生成")
                    
                    # 验证结果 - 应该正确解析数字
                    assert score == 0.85
    
    @pytest.mark.asyncio
    async def test_evaluate_answer_with_exception_handling(self):
        """
        测试异常处理 - 验证方法执行流程
        """
        with patch('service.evaluation.service.ChatOpenAI', return_value=MagicMock()):
            with patch('service.evaluation.service.ChatPromptTemplate.from_messages', return_value=MagicMock()):
                # 创建评估器实例
                evaluator = SimplePromptEvaluator("fake_api_key", "https://fake-api.com", "test-model")
                
                # 模拟抛出异常
                with patch.object(evaluator, 'chain', MagicMock()) as mock_chain:
                    mock_chain.ainvoke = AsyncMock(side_effect=Exception("测试异常"))
                    
                    # 调用评估方法 - 即使抛出异常也不应该失败
                    score = await evaluator.evaluate_answer("测试问题", "测试参考", "测试生成")
                    
                    # 验证结果 - 应该返回默认值0.5
                    assert score == 0.5
    
    @pytest.mark.asyncio
    async def test_aclose(self):
        """
        测试关闭异步资源
        """
        # 模拟LLM客户端
        mock_llm = MagicMock()
        mock_llm.client = MagicMock()
        mock_llm.client.aclose = AsyncMock()
        
        # 模拟客户端创建
        with patch('service.evaluation.service.ChatOpenAI', return_value=mock_llm):
            with patch('service.evaluation.service.ChatPromptTemplate.from_messages', return_value=MagicMock()):
                evaluator = SimplePromptEvaluator("fake_api_key", "https://fake-api.com", "test-model")
                
                # 调用关闭方法
                await evaluator.aclose()
                
                # 验证客户端关闭被调用 - 注意实际实现可能不同
                if hasattr(mock_llm.client, 'aclose'):
                    mock_llm.client.aclose.assert_called_once()
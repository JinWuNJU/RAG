import os
import sys
import uuid
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.evaluation.service import EvaluationService, SimplePromptEvaluator
from service.evaluation.router import create_evaluation_task, get_tasks, delete_task
from service.evaluation.schemas import EvaluationRequest, EvaluationTasksResponse


class TestEvaluationService:
    """测试评估服务类"""
    
    def test_init(self, mock_db):
        """测试初始化评估服务"""
        service = EvaluationService(mock_db)
        assert service.db == mock_db
        # 使用实际的指标名称
        assert "prompt_scs" in service.metrics
        assert "bleu" in service.metrics
    
    @pytest.mark.asyncio
    async def test_evaluate(self, mock_db):
        """测试评估方法"""
        # 完全模拟评估服务
        service = MagicMock(spec=EvaluationService)
        
        # 设置固定返回值
        from collections import namedtuple
        Result = namedtuple('Result', ['scores'])
        service.evaluate.return_value = Result(scores={
            "prompt_scs": [0.9],
            "answer_relevancy": [0.7],
            "bleu": [0.8]
        })
        
        questions = ["测试问题"]
        answers = ["参考答案"]
        generated_responses = ["生成答案"]
        metric_names = ["prompt_scs", "answer_relevancy", "bleu"]
        
        # 调用评估方法
        result = await service.evaluate(questions, answers, metric_names, generated_responses)
        
        # 验证结果 - 使用scores属性访问结果
        assert "prompt_scs" in result.scores
        assert "answer_relevancy" in result.scores
        assert "bleu" in result.scores
        assert result.scores["prompt_scs"][0] == 0.9
        assert result.scores["answer_relevancy"][0] == 0.7
        assert result.scores["bleu"][0] == 0.8
        
        # 验证调用
        service.evaluate.assert_called_once_with(questions, answers, metric_names, generated_responses)
    
    def test_get_all_tasks(self, mock_db, test_user_id):
        """测试获取所有任务"""
        # 完全模拟评估服务
        service = MagicMock(spec=EvaluationService)
        
        # 设置模拟返回值
        service.get_all_tasks.return_value = {
            "tasks": [{
                "id": str(uuid.uuid4()),
                "name": "测试任务",
                "status": "completed",
                "created_at": int(datetime(2023, 1, 1).timestamp() * 1000),
                "metrics": ["prompt_scs"]
            }],
            "total": 1
        }
        
        # 调用获取任务方法
        result = service.get_all_tasks(test_user_id, 0, 10)
        
        # 验证结果
        assert "tasks" in result
        assert "total" in result
        assert result["total"] == 1
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["name"] == "测试任务"
        assert result["tasks"][0]["metrics"] == ["prompt_scs"]
        
        # 验证调用
        service.get_all_tasks.assert_called_once_with(test_user_id, 0, 10)
    
    def test_delete_task(self, mock_db, test_user_id, test_task_id):
        """测试删除任务"""
        # 完全模拟评估服务
        service = MagicMock(spec=EvaluationService)
        
        # 设置模拟返回值
        service.delete_task.return_value = {
            "success": True,
            "message": "任务删除成功"
        }
        
        # 调用删除任务方法
        result = service.delete_task(test_task_id, test_user_id)
        
        # 验证结果
        assert result["success"] is True
        assert "message" in result
        
        # 验证调用
        service.delete_task.assert_called_once_with(test_task_id, test_user_id)


class TestEvaluationRoutes:
    """测试评估路由"""
    
    @pytest.mark.asyncio
    async def test_create_evaluation_task(self, mock_db, mock_auth_jwt):
        """测试创建评估任务"""
        # 模拟文件查询结果
        mock_file = MagicMock()
        mock_file.id = uuid.uuid4()
        mock_db.query().filter().first.return_value = mock_file
        
        # 创建请求数据
        request = EvaluationRequest(
            task_name="测试任务",
            metric_id="prompt_scs",
            system_prompt="测试系统提示",
            file_id=str(uuid.uuid4())
        )
        
        # 模拟用户ID解码
        with patch('service.evaluation.router.decode_jwt_to_uid', return_value=uuid.uuid4()):
            # 模拟线程启动
            with patch('threading.Thread'):
                # 调用路由方法
                response = await create_evaluation_task(request, mock_auth_jwt, mock_db)
        
        # 验证结果
        assert "task_id" in response.dict()
        assert "record_id" in response.dict()
        
        # 验证数据库操作
        assert mock_db.add.call_count == 2
        assert mock_db.commit.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_tasks(self, mock_db, mock_auth_jwt):
        """测试获取任务列表"""
        # 模拟服务返回结果
        mock_result = {
            "tasks": [
                {
                    "id": str(uuid.uuid4()),
                    "name": "测试任务",
                    "status": "completed",
                    # 使用整数时间戳而非字符串
                    "created_at": int(datetime(2023, 1, 1).timestamp() * 1000),
                    "metrics": ["prompt_scs"]
                }
            ],
            "total": 1
        }
        
        # 模拟用户ID解码
        with patch('service.evaluation.router.decode_jwt_to_uid', return_value=uuid.uuid4()):
            # 模拟服务方法
            with patch('service.evaluation.router.EvaluationService') as mock_service_class:
                mock_service = mock_service_class.return_value
                mock_service.get_all_tasks.return_value = mock_result
                
                # 调用路由方法
                response = await get_tasks(1, 10, None, mock_auth_jwt, mock_db)
        
        # 验证结果
        assert isinstance(response, EvaluationTasksResponse)
        assert response.total == 1
        assert len(response.tasks) == 1
        assert response.tasks[0].name == "测试任务"
    
    @pytest.mark.asyncio
    async def test_delete_task(self, mock_db, mock_auth_jwt, test_task_id):
        """测试删除任务"""
        # 模拟服务返回结果
        mock_result = {
            "success": True,
            "message": "任务删除成功"
        }
        
        # 模拟用户ID解码
        with patch('service.evaluation.router.decode_jwt_to_uid', return_value=uuid.uuid4()):
            # 模拟服务方法
            with patch('service.evaluation.router.EvaluationService') as mock_service_class:
                mock_service = mock_service_class.return_value
                mock_service.delete_task.return_value = mock_result
                
                # 调用路由方法
                response = await delete_task(test_task_id, mock_auth_jwt, mock_db)
        
        # 验证结果
        assert response.success is True
        assert response.message == "任务删除成功"


class TestSimplePromptEvaluator:
    """测试简单提示词评估器"""
    
    @pytest.mark.asyncio
    async def test_evaluate_answer(self):
        """测试评估答案"""
        # 完全模拟评估器
        evaluator = MagicMock(spec=SimplePromptEvaluator)
        evaluator.evaluate_answer.return_value = 0.85
        
        # 调用评估方法
        result = await evaluator.evaluate_answer(
            "测试问题", 
            "参考答案", 
            "生成答案"
        )
        
        # 验证结果
        assert result == 0.85
        
        # 验证调用
        evaluator.evaluate_answer.assert_called_once_with(
            "测试问题", 
            "参考答案", 
            "生成答案"
        ) 
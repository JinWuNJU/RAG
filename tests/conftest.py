import os
import sys
import uuid
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi_jwt_auth2 import AuthJWT
from sqlalchemy.orm import Session

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.model.evaluation import EvaluationTask, EvaluationRecord, CustomMetric


@pytest.fixture
def mock_db():
    """模拟数据库会话"""
    mock_session = MagicMock(spec=Session)
    
    # 模拟查询构建器
    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.filter_by.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.options.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.offset.return_value = mock_query
    mock_query.limit.return_value = mock_query
    
    return mock_session


@pytest.fixture
def mock_auth_jwt():
    """模拟JWT认证"""
    mock_auth = MagicMock(spec=AuthJWT)
    mock_auth.get_jwt_subject.return_value = str(uuid.uuid4())
    return mock_auth


@pytest.fixture(scope="function")
def test_user_id():
    """测试用户ID"""
    return uuid.uuid4()


@pytest.fixture(scope="function")
def real_user_id():
    """实际存在的用户ID"""
    return uuid.UUID("f97f4311-987d-4867-af62-2fd3a2b80992")


@pytest.fixture(scope="function")
def test_task_id():
    """测试任务ID"""
    return uuid.uuid4()


@pytest.fixture(scope="function")
def test_record_id():
    """测试记录ID"""
    return uuid.uuid4()


@pytest.fixture(scope="function")
def test_file_id():
    """测试文件ID"""
    return uuid.uuid4()


@pytest.fixture
def test_evaluation_task(test_user_id):
    """创建测试评估任务"""
    return EvaluationTask(
        id=uuid.uuid4(),
        name="测试评估任务",
        user_id=test_user_id,
        status="processing",
        created_at="2023-01-01T00:00:00"
    )


@pytest.fixture
def test_evaluation_record(test_task_id, test_file_id):
    """创建测试评估记录"""
    return EvaluationRecord(
        id=uuid.uuid4(),
        task_id=test_task_id,
        metric_id="accuracy",
        system_prompt="测试系统提示",
        file_id=test_file_id,
        created_at="2023-01-01T00:00:00"
    )


@pytest.fixture
def test_custom_metric(test_user_id):
    """创建测试自定义指标"""
    return CustomMetric(
        id=uuid.uuid4(),
        user_id=test_user_id,
        name="测试自定义指标",
        description="测试自定义指标描述",
        criteria=[{"name": "准确性", "weight": 0.5}],
        instruction="评估指令",
        scale=10,
        type="custom",
        created_at="2023-01-01T00:00:00"
    )


@pytest.fixture
def mock_evaluation_service():
    """模拟评估服务"""
    with patch('service.evaluation.service.EvaluationService') as mock:
        service_instance = mock.return_value
        
        # 模拟评估方法，使用固定值与测试保持一致
        service_instance.evaluate.return_value = {
            "accuracy": 0.85,
            "relevancy": 0.88,
            "completeness": 0.91
        }
        
        # 模拟RAG评估方法，返回固定值与测试保持一致
        from collections import namedtuple
        Result = namedtuple('Result', ['scores'])
        service_instance.evaluate_rag.return_value = Result(scores={
            "faithfulness": [0.87],
            "context_relevancy": [0.88],
            "context_precision": [0.91]
        })
        
        # 模拟特定评估方法
        service_instance._evaluate_faithfulness = AsyncMock(return_value=[0.87])
        service_instance._evaluate_context_relevancy = AsyncMock(return_value=[0.88])
        service_instance._evaluate_context_precision = AsyncMock(return_value=[0.91])
        
        # 模拟获取任务方法
        service_instance.get_all_tasks.return_value = {
            "tasks": [
                {
                    "id": str(uuid.uuid4()),
                    "name": "测试任务",
                    "status": "completed",
                    "created_at": "2023-01-01T00:00:00",
                    "metrics": ["accuracy"]
                }
            ],
            "total": 1
        }
        
        # 模拟获取任务记录方法
        service_instance.get_task_records.return_value = [
            {
                "id": str(uuid.uuid4()),
                "task_id": str(uuid.uuid4()),
                "metric_id": "accuracy",
                "system_prompt": "测试系统提示",
                "file_id": str(uuid.uuid4()),
                "created_at": "2023-01-01T00:00:00",
                "results": {"accuracy": 0.85}
            }
        ]
        
        yield service_instance 
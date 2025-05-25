import pytest
from fastapi import HTTPException
from uuid import UUID, uuid4
from unittest.mock import MagicMock, AsyncMock, patch, call
from sqlalchemy.orm import Session

# 被测模块
from service.knowledge_base import knowledge_base_router  as kb_router
from rest_model.knowledge_base import *
from database.model.knowledge_base import KnowledgeBase, KnowledgeBaseChunk
from database.model.user import User
from service.user.auth import decode_jwt_to_uid
# 测试常量
TEST_USER_ID = uuid4()
FILE_ID = uuid4()
OTHER_USER_ID = uuid4()
TEST_KB_ID = uuid4()
MAX_KB_PER_USER = 10


@pytest.fixture
def mock_db():
    """增强版数据库模拟"""
    db = MagicMock(spec=Session)

    # 创建独立的查询对象mock
    query_mock = MagicMock()
    query_mock.filter.return_value = query_mock
    query_mock.filter_by.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.offset.return_value = query_mock
    query_mock.limit.return_value = query_mock
    query_mock.first.return_value = None  # 默认返回None

    # 配置Session.query()返回查询mock
    db.query.return_value = query_mock

    # 模拟数据库操作
    added_objects = []

    def add_mock(obj):
        nonlocal added_objects
        added_objects.append(obj)
        # 为KnowledgeBase对象生成模拟ID
        if isinstance(obj, KnowledgeBase):
            obj.id = uuid4()  # 动态生成UUID

    def refresh_mock(obj):
        # 模拟刷新后保持ID
        if isinstance(obj, KnowledgeBase) and not obj.id:
            obj.id = uuid4()

    # 绑定模拟方法
    db.add = MagicMock(side_effect=add_mock)
    db.commit = MagicMock()
    db.refresh = MagicMock(side_effect=refresh_mock)

    return db


@pytest.fixture
def mock_auth():
    """模拟JWT认证"""
    auth = MagicMock()
    auth.decode_jwt_to_uid.return_value = TEST_USER_ID
    return auth


@patch("service.user.auth.decode_jwt_to_uid")  # 修正后的正确路径
async def test_create_kb_success(mock_decode_jwt, mock_db):
    """测试成功创建知识库"""
    # 配置模拟用户数据
    mock_user = User(id=TEST_USER_ID, knowledge_base_count=5)
    mock_db.query().filter().first.return_value = mock_user

    # 配置模拟请求数据
    test_data = KnowledgeBaseCreate(
        name="Test KB",
        description="Test Description",
        file_ids=[FILE_ID],
        chunk_size=512,
        overlap_size=128,
        is_public=True
    )

    # 执行API调用
    result = await kb_router.create_knowledge_base(
        data=test_data,
        background_tasks=MagicMock(),
        db=mock_db,
        Authorize=MagicMock()
    )

    # 验证结果
    assert isinstance(result["knowledge_base_id"], UUID)
    assert result["status"] == "building"

    # 验证用户知识库计数更新
    assert mock_user.knowledge_base_count == 6



@patch("service.user.auth.decode_jwt_to_uid")
async def test_create_kb_limit_exceeded(mock_auth, mock_db):
    """测试超过知识库数量限制"""
    mock_user = User(id=TEST_USER_ID, knowledge_base_count=MAX_KB_PER_USER)
    mock_db.query().filter().first.return_value = mock_user

    with pytest.raises(HTTPException) as exc:
        await kb_router.create_knowledge_base(
            data=KnowledgeBaseCreate(name="Test KB", file_ids=[]),
            background_tasks=MagicMock(),
            db=mock_db,
            Authorize=MagicMock()
        )

    assert exc.value.status_code == 400
    assert f"最多只能创建{MAX_KB_PER_USER}个" in exc.value.detail


@patch("service.user.auth.decode_jwt_to_uid")
async def test_list_knowledge_bases_success(mock_decode_jwt, mock_db):
    """测试成功获取知识库列表"""
    # 模拟数据
    mock_user = User(id=TEST_USER_ID)
    public_kb = KnowledgeBase(
        id=TEST_KB_ID,
        name="Public KB",
        is_public=True,
        uploader_id=OTHER_USER_ID,
        created_at=datetime.now(),
        status="completed"  # 新增必填字段
    )
    private_kb = KnowledgeBase(
        id=uuid4(),
        name="Private KB",
        is_public=False,
        uploader_id=TEST_USER_ID,
        created_at=datetime.now(),
        status="building"  # 新增必填字段
    )

    # 配置查询返回值
    mock_db.query().filter().order_by().offset().limit().all.return_value = [public_kb, private_kb]
    mock_db.query().filter().count.return_value = 2

    result = await kb_router.list_knowledge_bases(
        db=mock_db,
        Authorize=MagicMock(),
        page=0,
        limit=10
    )

    assert result.total == 2
    assert len(result.items) == 2



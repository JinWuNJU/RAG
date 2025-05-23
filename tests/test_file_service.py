import pytest
from fastapi import UploadFile, HTTPException
from starlette.datastructures import Headers
from uuid import UUID
import io
from typing import Generator
from sqlalchemy.orm import Session
from unittest.mock import MagicMock, AsyncMock, patch
import datetime

from service.file.file import (
    upload_to_database,
    get_file_by_id,
    get_file_metadata,
    delete_file,
    get_user_files,
    get_user_file_count,
    FileCountLimitError
)
from service.file.config import MAX_FILES_PER_USER
from database.model.file import FileDB
from rest_model.file import FileTypeError, FileSizeError, FileMetadata

# 测试用户ID
TEST_USER_ID = UUID('12345678-1234-5678-1234-567812345678')

class MockUploadFile(UploadFile):
    def __init__(self, filename: str, content: bytes, content_type: str):
        self._content = content  # 存储内容以便后续使用
        super().__init__(
            file=io.BytesIO(content),
            filename=filename,
            headers=Headers({"content-type": content_type})
        )
        self.file = io.BytesIO(content)  # 确保文件指针被正确设置

    async def read(self, size: int | None = None) -> bytes:
        """重写read方法以返回完整内容"""
        if size is None:
            return self._content
        return self._content[:size]

    async def seek(self, offset: int) -> None:
        """重写seek方法"""
        self.file.seek(offset)

@pytest.fixture
def mock_file() -> Generator[UploadFile, None, None]:
    """创建模拟文件对象"""
    content = b"Test file content"
    file = MockUploadFile(
        filename="test.txt",
        content=content,
        content_type="text/plain"
    )
    yield file
    file.file.close()

@pytest.fixture
def large_mock_file() -> Generator[UploadFile, None, None]:
    """创建大文件对象（2MB）"""
    content = b"0" * (2 * 1024 * 1024)  # 2MB
    file = MockUploadFile(
        filename="large_test.txt",
        content=content,
        content_type="text/plain"
    )
    yield file
    file.file.close()

@pytest.fixture
def mock_db():
    """创建模拟数据库会话"""
    db = MagicMock(spec=Session)
    
    # 模拟查询构建器
    mock_query = MagicMock()
    db.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.filter_by.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.options.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.offset.return_value = mock_query
    mock_query.limit.return_value = mock_query
    
    return db

@patch("service.file.file.get_user_file_count", new_callable=AsyncMock)
async def test_upload_file_success(mock_count, mock_db: Session, mock_file: UploadFile):
    """测试文件上传成功"""
    mock_count.return_value = 0
    mock_db.add = MagicMock()
    mock_db.commit = MagicMock()
    # patch refresh 让其给 db_file.id 赋值
    def refresh_side_effect(db_file):
        db_file.id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    mock_db.refresh = MagicMock(side_effect=refresh_side_effect)

    # 重置文件指针
    await mock_file.seek(0)

    file_id = await upload_to_database(
        file=mock_file,
        db=mock_db,
        user_id=TEST_USER_ID
    )

    assert isinstance(file_id, UUID)
    assert file_id == UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once()

@patch("service.file.file.get_user_file_count", new_callable=AsyncMock)
async def test_file_size_limit(mock_count, mock_db: Session, large_mock_file: UploadFile):
    """测试文件大小限制"""
    mock_count.return_value = 0
    with pytest.raises(FileSizeError):
        await upload_to_database(
            file=large_mock_file,
            db=mock_db,
            user_id=TEST_USER_ID,
            max_size_mb=1  # 1MB限制
        )

@patch("service.file.file.get_user_file_count", new_callable=AsyncMock)
async def test_file_type_restriction(mock_count, mock_db: Session, mock_file: UploadFile):
    """测试文件类型限制"""
    mock_count.return_value = 0
    with pytest.raises(FileTypeError):
        await upload_to_database(
            file=mock_file,
            db=mock_db,
            user_id=TEST_USER_ID,
            allowed_types=["application/pdf"]  # 只允许PDF
        )

@patch("service.file.file.get_user_file_count", new_callable=AsyncMock)
async def test_file_count_limit(mock_count, mock_db: Session, mock_file: UploadFile):
    """测试文件数量限制"""
    mock_count.return_value = MAX_FILES_PER_USER
    with pytest.raises(FileCountLimitError):
        await upload_to_database(
            file=mock_file,
            db=mock_db,
            user_id=TEST_USER_ID
        )

async def test_get_file_metadata(mock_db: Session, mock_file: UploadFile):
    """测试获取文件元数据"""
    # 模拟数据库中的文件记录
    now = datetime.datetime.now()
    mock_file_db = FileDB(
        id=UUID('11111111-1111-1111-1111-111111111111'),
        filename="test.txt",
        content_type="text/plain",
        size=100,
        user_id=TEST_USER_ID,
        is_public=False,
        created_at=now,
        updated_at=now
    )
    mock_db.query().filter().first.return_value = mock_file_db
    
    metadata = await get_file_metadata(mock_file_db.id, mock_db, TEST_USER_ID)
    assert isinstance(metadata, FileMetadata)
    assert metadata.filename == "test.txt"
    assert metadata.content_type == "text/plain"
    assert metadata.id == mock_file_db.id

async def test_delete_file(mock_db: Session, mock_file: UploadFile):
    """测试删除文件"""
    now = datetime.datetime.now()
    # 模拟数据库中的文件记录
    mock_file_db = FileDB(
        id=UUID('11111111-1111-1111-1111-111111111111'),
        filename="test.txt",
        content_type="text/plain",
        size=100,
        user_id=TEST_USER_ID,
        is_public=False,
        created_at=now,
        updated_at=now
    )
    mock_db.query().filter().first.return_value = mock_file_db
    mock_db.delete = MagicMock()
    mock_db.commit = MagicMock()
    
    success = await delete_file(mock_file_db.id, mock_db, TEST_USER_ID)
    assert success is True
    mock_db.delete.assert_called_once_with(mock_file_db)
    mock_db.commit.assert_called_once()

async def test_get_user_files(mock_db: Session, mock_file: UploadFile):
    """测试获取用户文件列表"""
    now = datetime.datetime.now()
    # 模拟数据库中的文件记录列表
    mock_files = [
        FileDB(
            id=UUID('11111111-1111-1111-1111-111111111111'),
            filename="test1.txt",
            content_type="text/plain",
            size=100,
            user_id=TEST_USER_ID,
            is_public=False,
            created_at=now,
            updated_at=now
        ),
        FileDB(
            id=UUID('22222222-2222-2222-2222-222222222222'),
            filename="test2.txt",
            content_type="text/plain",
            size=200,
            user_id=TEST_USER_ID,
            is_public=False,
            created_at=now,
            updated_at=now
        )
    ]
    mock_db.query().filter().all.return_value = mock_files
    
    files = await get_user_files(mock_db, TEST_USER_ID)
    assert len(files) == 2
    assert all(isinstance(f, FileMetadata) for f in files)
    assert files[0].filename == "test1.txt"
    assert files[1].filename == "test2.txt"

async def test_get_user_file_count(mock_db: Session, mock_file: UploadFile):
    """测试获取用户文件数量"""
    # 模拟当前文件数量
    mock_db.query().scalar.return_value = 5
    
    count = await get_user_file_count(mock_db, TEST_USER_ID)
    assert count == 5 
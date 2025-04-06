from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import io
from fastapi_jwt_auth2 import AuthJWT

from rest_model.file import FileUploadResponse, FileMetadata, FileTypeError, FileSizeError
from database import get_db
from service.file.file import upload_to_database, get_file_by_id, get_file_metadata, delete_file, get_user_files
from service.user import auth

router = APIRouter(tags=["files"], prefix="/files")

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    allowed_types: Optional[List[str]] = Form(None),
    max_size_mb: Optional[float] = Form(None),
    is_public: bool = Form(False),
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends()
):
    """
    通用文件上传接口
    
    参数:
        file: 上传的文件
        allowed_types: 允许的文件类型列表 (MIME类型)
        max_size_mb: 最大文件大小 (MB)
        is_public: 文件是否公开可访问
        
    返回:
        文件ID
    """
    try:
        file_id = await upload_to_database(
            file=file,
            db=db,
            allowed_types=allowed_types,
            max_size_mb=max_size_mb,
            is_public=is_public,
            user_id=auth.decode_jwt_to_uid(Authorize)
        )

        return FileUploadResponse(file_id=file_id)
        
    except FileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileSizeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{file_id}/metadata", response_model=FileMetadata)
async def get_file_info(
    file_id: UUID, 
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends()
):
    """
    获取文件元数据
    
    参数:
        file_id: 文件ID
        
    返回:
        文件元数据
    """
    metadata = await get_file_metadata(file_id, db, auth.decode_jwt_to_uid(Authorize))
    return metadata

@router.get("/{file_id}")
async def download_file(
    file_id: UUID, 
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends()
):
    """
    下载文件
    
    参数:
        file_id: 文件ID
        
    返回:
        文件内容
    """
    
    file = await get_file_by_id(file_id, db, auth.decode_jwt_to_uid(Authorize))
    
    return StreamingResponse(
        io.BytesIO(file.data),
        media_type=file.content_type,
        headers={"Content-Disposition": f"attachment; filename={file.filename}"}
    )

@router.delete("/{file_id}")
async def remove_file(
    file_id: UUID, 
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends()
):
    """
    删除文件
    
    参数:
        file_id: 文件ID
        
    返回:
        删除成功状态
    """
    
    success = await delete_file(file_id, db, auth.decode_jwt_to_uid(Authorize))
    
    if success:
        return True
    else:
        raise HTTPException(status_code=500, detail="Failed to delete file")

# 这些是特定用途的示例接口，便于其他开发者使用
@router.post("/prompt/eval_data/upload", response_model=FileUploadResponse)
async def upload_eval_data(
    file: UploadFile = File(...),
    max_size_mb: float = Form(10.0),
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends()
):
    """
    上传评估数据文件（JSON格式）
    
    参数:
        file: 上传的文件（JSON格式）
        max_size_mb: 最大文件大小（默认10MB）
        
    返回:
        文件ID
    """
    
    try:
        file_id = await upload_to_database(
            file=file,
            db=db,
            allowed_types=["application/json"],
            max_size_mb=max_size_mb,
            user_id=auth.decode_jwt_to_uid(Authorize)
        )

        return FileUploadResponse(file_id=file_id)
        
    except FileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileSizeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/prompt/knowledge/upload", response_model=FileUploadResponse)
async def upload_knowledge(
    file: UploadFile = File(...),
    max_size_mb: float = Form(50.0),
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends()
):
    """
    上传知识库文件（纯文本格式）
    
    参数:
        file: 上传的文件（纯文本格式）
        max_size_mb: 最大文件大小（默认50MB）
        
    返回:
        文件ID
    """
    
    try:
        file_id = await upload_to_database(
            file=file,
            db=db,
            allowed_types=["text/plain"],
            max_size_mb=max_size_mb,
            user_id=auth.decode_jwt_to_uid(Authorize)
        )
        
        return FileUploadResponse(file_id=file_id)
        
    except FileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileSizeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[FileMetadata])
async def list_my_files(
    include_public: bool = Query(False, description="包含所有公开文件"),
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends()
):
    """
    获取当前用户的所有文件
    
    参数:
        include_public: 是否包含公开文件，默认为False
        
    返回:
        文件元数据列表
    """
    
    return await get_user_files(db, auth.decode_jwt_to_uid(Authorize), include_public) 
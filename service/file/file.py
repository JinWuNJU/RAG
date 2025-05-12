from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
from sqlalchemy import or_

from database.model.file import *
from rest_model.file import FileMetadata, FileTypeError, FileSizeError

async def upload_to_database(
    file: UploadFile, 
    db: Session,
    user_id: UUID,
    allowed_types: Optional[List[str]] = None,
    max_size_mb: Optional[float] = None,
    is_public: bool = False
) -> UUID:
    """
    Upload a file to the database
    
    Args:
        file: The uploaded file
        db: Database session
        allowed_types: List of allowed MIME types. If None, all types are allowed
        max_size_mb: Maximum file size in MB. If None, no limit
        user_id
        is_public: Whether the file is publicly accessible
        
    Returns:
        UUID of the uploaded file
    """
    if file.content_type is None:
        raise FileTypeError("File type is missing")
    
    if allowed_types and file.content_type not in allowed_types:
        raise FileTypeError(f"File type {file.content_type} not allowed. Allowed types: {', '.join(allowed_types)}")
    
    content = await file.read()
    file_size = len(content)
    
    if max_size_mb and file_size > max_size_mb * 1024 * 1024:
        raise FileSizeError(f"File size exceeds maximum allowed size of {max_size_mb} MB")
    
    if file.filename is None:
        raise FileTypeError("File name is missing")

    db_file = FileDB()
    db_file.filename = file.filename
    db_file.content_type = file.content_type
    db_file.size = file_size
    db_file.data = content
    db_file.user_id = user_id
    db_file.is_public = is_public
    
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    
    return db_file.id

async def get_file_by_id(file_id: UUID, db: Session, user_id: UUID) -> FileDB:
    """
    Get a file by its ID
    
    Args:
        file_id: UUID of the file
        db: Database session
        user_id: UUID of user
        
    Returns:
        FileDB object
        
    Raises:
        HTTPException: If file not found or permission denied
    """
    file = db.query(FileDB).filter(FileDB.id == file_id).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file.is_public and file.user_id:
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required to access this file")
        
        if not user_id or user_id != file.user_id:
            raise HTTPException(status_code=403, detail="Permission denied to access this file")
    
    return file

async def get_file_metadata(file_id: UUID, db: Session, user_id: UUID) -> FileMetadata:
    """
    Get file metadata without the actual file content
    
    Args:
        file_id: UUID of the file
        db: Database session
        user_id: UUID of user
        
    Returns:
        FileMetadata object
        
    Raises:
        HTTPException: If file not found or permission denied
    """
    file = await get_file_by_id(file_id, db, user_id)
    
    return FileMetadata(
        id=file.id,
        filename=file.filename,
        content_type=file.content_type,
        size=file.size,
        is_public=file.is_public,
        created_at=file.created_at,
        updated_at=file.updated_at
    )

async def delete_file(file_id: UUID, db: Session, user_id: UUID) -> bool:
    """
    Delete a file by its ID
    
    Args:
        file_id: UUID of the file
        db: Database session
        user_id: UUID of the user requesting deletion
        
    Returns:
        True if successful
        
    Raises:
        HTTPException: If file not found or permission denied
    """
    file = await get_file_by_id(file_id, db, user_id)
    
    db.delete(file)
    db.commit()
    
    return True

async def get_user_files(db: Session, user_id: UUID, include_public: bool = False):
    """
    Get all files for a user
    
    Args:
        db: Database session
        user_id: UUID of the user
        include_public: Whether to include public files
        
    Returns:
        List of FileMetadata objects
    """
    if include_public:
        files = db.query(FileDB).filter(
            or_(
                FileDB.user_id == user_id,
                FileDB.is_public == True
            )
        ).all()
    else:
        files = db.query(FileDB).filter(FileDB.user_id == user_id).all()
    
    result = []
    for file in files:
        result.append(
            FileMetadata(
                id=file.id,
                filename=file.filename,
                content_type=file.content_type,
                size=file.size,
                is_public=file.is_public,
                created_at=file.created_at,
                updated_at=file.updated_at
            )
        )
    
    return result

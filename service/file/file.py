from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID

from model.file.file import FileDB, FileMetadata, FileTypeError, FileSizeError
from service.database import engine

from model.file.file import Base
Base.metadata.create_all(bind=engine)

async def upload_to_database(
    file: UploadFile, 
    db: Session,
    allowed_types: Optional[List[str]] = None,
    max_size_mb: Optional[float] = None,
    user_id: Optional[UUID] = None,
    is_public: bool = False
) -> UUID:
    """
    Upload a file to the database
    
    Args:
        file: The uploaded file
        db: Database session
        allowed_types: List of allowed MIME types. If None, all types are allowed
        max_size_mb: Maximum file size in MB. If None, no limit
        user_id: User ID of the uploader (optional)
        is_public: Whether the file is publicly accessible
        
    Returns:
        UUID of the uploaded file
    """
    if allowed_types and file.content_type not in allowed_types:
        raise FileTypeError(f"File type {file.content_type} not allowed. Allowed types: {', '.join(allowed_types)}")
    
    content = await file.read()
    file_size = len(content)
    
    if max_size_mb and file_size > max_size_mb * 1024 * 1024:
        raise FileSizeError(f"File size exceeds maximum allowed size of {max_size_mb} MB")
    
    db_file = FileDB(
        filename=file.filename,
        content_type=file.content_type,
        size=file_size,
        data=content,
        user_id=user_id,
        is_public=is_public
    )
    
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    
    return db_file.id

async def get_file_by_id(file_id: UUID, db: Session) -> FileDB:
    """
    Get a file by its ID
    
    Args:
        file_id: UUID of the file
        db: Database session
        
    Returns:
        FileDB object
        
    Raises:
        HTTPException: If file not found
    """
    file = db.query(FileDB).filter(FileDB.id == file_id).first()
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    return file

async def get_file_metadata(file_id: UUID, db: Session) -> FileMetadata:
    """
    Get file metadata without the actual file content
    
    Args:
        file_id: UUID of the file
        db: Database session
        
    Returns:
        FileMetadata object
        
    Raises:
        HTTPException: If file not found
    """
    file = await get_file_by_id(file_id, db)
    
    return FileMetadata(
        id=file.id,
        filename=file.filename,
        content_type=file.content_type,
        size=file.size,
        is_public=file.is_public,
        created_at=file.created_at,
        updated_at=file.updated_at
    )

async def delete_file(file_id: UUID, db: Session, user_id: Optional[UUID] = None) -> bool:
    """
    Delete a file by its ID
    
    Args:
        file_id: UUID of the file
        db: Database session
        user_id: User ID of the requester (optional, for permission check)
        
    Returns:
        True if successful
        
    Raises:
        HTTPException: If file not found or permission denied
    """
    file = await get_file_by_id(file_id, db)
    
    if user_id and file.user_id and file.user_id != user_id and not file.is_public:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    db.delete(file)
    db.commit()
    
    return True 
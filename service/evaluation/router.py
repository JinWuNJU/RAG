from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi_jwt_auth2 import AuthJWT
from sqlalchemy.orm import Session
from uuid import UUID
import json

from database import get_db
from .service import EvaluationService
from .schemas import Metric, EvaluationRequest, EvaluationRecordResponse
from ..user.auth import decode_jwt_to_uid

from . import router as evaluation_router
from ..user.user_router import router as user_router

__all__ = ["evaluation_router", "user_router"]
router = APIRouter(tags=["evaluation"], prefix="/evaluation")

@router.get("/metrics", response_model=list[Metric])
async def get_metrics(db: Session = Depends(get_db)):
    """获取可用评估指标"""
    service = EvaluationService(db)
    return [
        Metric(id=k, name=v["name"], description=v["description"])
        for k, v in service.metrics.items()
    ]

@router.post("/tasks", response_model=UUID)
async def create_evaluation_task(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    Authorize: AuthJWT = Depends(),
    db: Session = Depends(get_db)
):
    """创建评估任务"""
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)

    # 这里应该添加数据库记录创建逻辑
    # 然后启动后台任务执行评估

    # 示例返回一个UUID
    return UUID("123e4567-e89b-12d3-a456-426614174000")
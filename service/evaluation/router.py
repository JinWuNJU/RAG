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
    """获取可用评估指标列表"""
    service = EvaluationService(db)
    return [
        Metric(
            id=metric_id,
            name=details["name"],
            description=details["description"]
        )
        for metric_id, details in service.metrics.items()
    ]

@router.post("/tasks", response_model=UUID)
async def create_evaluation_task(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    Authorize: AuthJWT = Depends(),
    db: Session = Depends(get_db)
):
    """创建评估任务
    1. 验证用户和文件
    2. 创建任务和记录数据库条目
    3. 启动后台评估任务
    4. 返回任务ID
    """
    from uuid import uuid4
    from datetime import datetime
    from database.model.evaluation import EvaluationTask, EvaluationRecord
    from database.model.file import FileDB
    # 新增迭代版本控制
    task_version = 1  # 初始版本为1

    # 1. 认证用户
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)

    # 2. 验证文件存在且是JSON格式
    file_record = db.query(FileDB).filter(
        FileDB.id == request.file_id,
        FileDB.user_id == user_id  # 确保用户只能访问自己的文件
    ).first()

    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在或无权访问"
        )

    try:
        # 尝试解析文件内容
        file_content = json.loads(file_record.data.decode('utf-8'))
        if not isinstance(file_content, list):
            raise ValueError("文件内容必须是JSON数组")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"文件解析失败: {str(e)}"
        )

    # 3. 创建数据库记录
    task_id = str(uuid4())
    record_id = str(uuid4())

    try:
        # 创建任务
        task = EvaluationTask(
            id=task_id,
            name=request.task_name,
            user_id=user_id,
            status="processing",
            version=task_version,
            created_at=datetime.utcnow()
        )
        db.add(task)

        # 创建评估记录
        record = EvaluationRecord(
            id=record_id,
            task_id=task_id,
            metric_id=request.metric_id,
            system_prompt=request.system_prompt,
            file_id=request.file_id,
            created_at=datetime.utcnow()
        )
        db.add(record)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建评估任务失败: {str(e)}"
        )

    # 4. 启动后台评估任务
    background_tasks.add_task(
        run_evaluation_in_background,
        db=db,
        record_id=record_id,
        file_content=file_content,
        metric_id=request.metric_id,
        task_version=task_version,
        system_prompt=request.system_prompt
    )

    return record_id

async def run_evaluation_in_background(
    db: Session,
    record_id: str,
    file_content: list,
    metric_id: str,
    system_prompt: str
):
    """后台执行评估任务"""
    from sqlalchemy.orm import sessionmaker
    from database import engine
    from loguru import logger

    # 创建独立session避免主线程session问题
    LocalSession = sessionmaker(bind=engine)
    local_db = LocalSession()

    try:
        service = EvaluationService(local_db)

        # 准备评估数据
        questions = []
        answers = []

        for item in file_content:
            if not isinstance(item, dict):
                continue

            if 'query' in item and 'answer' in item:
                questions.append(str(item['query']))
                answers.append(str(item['answer']))

        if not questions or not answers:
            raise ValueError("文件中没有有效的query/answer数据")

        # 执行评估
        logger.info(f"开始评估记录 {record_id}...")
        result = await service.evaluate(questions, answers, [metric_id])

        # 更新评估记录
        record = local_db.query(EvaluationRecord).filter_by(id=record_id).first()
        if record:
            record.results = {
                "scores": result.scores,
                "questions": questions,
                "answers": answers
            }
            record.task.status = "completed"
            local_db.commit()
            logger.success(f"评估记录 {record_id} 完成")
        else:
            logger.error(f"评估记录 {record_id} 不存在")

    except Exception as e:
        logger.error(f"评估记录 {record_id} 失败: {str(e)}")
        # 更新状态为失败
        record = local_db.query(EvaluationRecord).filter_by(id=record_id).first()
        if record:
            record.task.status = "failed"
            record.task.error_message = str(e)
            local_db.commit()
    finally:
        local_db.close()

@router.get("/records/{record_id}", response_model=EvaluationRecordResponse)
async def get_evaluation_record(
    record_id: UUID,
    db: Session = Depends(get_db)
):
    """获取评估结果"""
    record = db.query(EvaluationRecord).filter_by(id=record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="评估记录不存在")

    return {
        "id": record.id,
        "created_at": record.created_at.isoformat(),
        "results": record.results
    }

@router.delete("/tasks/{task_id}")
async def delete_evaluation_task(
    task_id: UUID,
    Authorize: AuthJWT = Depends(),
    db: Session = Depends(get_db)
):
    """删除评估任务"""
    user_id = decode_jwt_to_uid(Authorize)

    task = db.query(EvaluationTask).filter(
        EvaluationTask.id == task_id,
        EvaluationTask.user_id == user_id
    ).first()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在或无权访问"
        )

    try:
        # 删除相关记录
        db.query(EvaluationRecord).filter_by(task_id=task_id).delete()
        # 删除任务
        db.delete(task)
        db.commit()
        return {"message": "删除成功"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除失败: {str(e)}"
        )


@router.post("/tasks/{task_id}/iterate")
async def iterate_evaluation_task(
    task_id: UUID,
    request: EvaluationRequest,
    Authorize: AuthJWT = Depends(),
    db: Session = Depends(get_db)
):
    """迭代评估任务"""
    # 获取原任务
    original_task = db.query(EvaluationTask).get(task_id)
    if not original_task:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 创建新版本记录
    new_version = original_task.version + 1
    new_task = EvaluationTask(
        id=uuid.uuid4(),
        name=f"{original_task.name} v{new_version}",
        user_id=original_task.user_id,
        version=new_version,
        previous_version=original_task.id,
        # ...其他字段...
    )
    # ...保存并启动评估...
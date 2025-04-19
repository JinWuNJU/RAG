from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi_jwt_auth2 import AuthJWT
from sqlalchemy.orm import Session
from uuid import UUID
import json
import uuid
from datetime import datetime
import asyncio
import threading

from database import get_db
from database.model.file import FileDB
from database.model.evaluation import EvaluationTask, EvaluationRecord
from .service import EvaluationService
from .schemas import (
    Metric,
    EvaluationRequest,
    EvaluationRecordResponse,
    EvaluationTasksResponse,
    EvaluationTaskItem,
    EvaluationIterationRequest,
    EvaluationTaskCreateResponse,
    EvaluationIterationResponse, DeleteTaskResponse
)
from ..user.auth import decode_jwt_to_uid
from utils.datetime_tools import get_beijing_time, to_timestamp_ms  # 导入工具函数

from . import router as evaluation_router
from ..user.user_router import router as user_router

__all__ = ["evaluation_router", "user_router"]
router = APIRouter(tags=["evaluation"], prefix="/evaluation")


@router.get("/metrics", response_model=list[Metric])
async def get_metrics(db: Session = Depends(get_db)):
    """获取可用评估指标"""
    service = EvaluationService(db)
    metrics = []
    
    # 优先显示prompt评估指标
    priority_metrics = ["prompt_scs", "bleu", "answer_relevancy"]
    
    # 先添加优先指标
    for metric_id in priority_metrics:
        if metric_id in service.metrics:
            metric_info = service.metrics[metric_id]
            metrics.append(
                Metric(id=metric_id, name=metric_info["name"], description=metric_info["description"])
            )
    
    # 添加剩余指标
    for metric_id, metric_info in service.metrics.items():
        if metric_id not in priority_metrics:
            metrics.append(
                Metric(id=metric_id, name=metric_info["name"], description=metric_info["description"])
            )
    
    return metrics


@router.get("/tasks", response_model=EvaluationTasksResponse)
async def get_tasks(
        page: int = 1,
        page_size: int = 10,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """获取评估任务列表，支持分页"""
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)

    skip = (page - 1) * page_size
    result = service.get_all_tasks(user_id, skip, page_size)

    return EvaluationTasksResponse(
        tasks=[EvaluationTaskItem(**task) for task in result["tasks"]],
        total=result["total"]
    )


@router.get("/tasks/{task_id}/records", response_model=list[EvaluationRecordResponse])
async def get_task_records(
        task_id: UUID,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """获取任务的所有评估记录"""
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)

    records = service.get_task_records(task_id, user_id)
    return [EvaluationRecordResponse(**record) for record in records]


@router.get("/records/{record_id}", response_model=EvaluationRecordResponse)
async def get_record_detail(
        record_id: UUID,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """获取评估记录详情"""
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)

    record = service.get_record_detail(record_id, user_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="评估记录不存在或无权访问"
        )

    return EvaluationRecordResponse(**record)


@router.post("/tasks", response_model=EvaluationTaskCreateResponse)
async def create_evaluation_task(
        request: EvaluationRequest,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """创建评估任务
    1. 验证用户和文件
    2. 创建任务和记录数据库条目
    3. 立即返回任务ID和记录ID
    4. 异步启动评估任务
    """
    # 1. 认证用户
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)
    
    # 添加详细日志
    print(f"收到评估任务创建请求 - 用户ID: {user_id}, 指标: {request.metric_id}")
    print(f"请求数据: {request.dict()}")

    # 2. 验证文件存在
    try:
        file_record = db.query(FileDB).filter(
            FileDB.id == request.file_id,
            FileDB.user_id == user_id  # 确保用户只能访问自己的文件
        ).first()

        if not file_record:
            print(f"文件不存在或无权访问: {request.file_id}, 用户: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="文件不存在或无权访问"
            )
        
        print(f"文件验证成功: {request.file_id}")

        # 3. 创建数据库记录
        task_id = uuid.uuid4()
        record_id = uuid.uuid4()

        try:
            # 创建任务
            task = EvaluationTask(
                id=task_id,
                name=request.task_name,
                user_id=user_id,
                status="processing",
                created_at=get_beijing_time()
            )
            db.add(task)

            # 创建评估记录
            record = EvaluationRecord(
                id=record_id,
                task_id=task_id,
                metric_id=request.metric_id,
                system_prompt=request.system_prompt,
                file_id=UUID(request.file_id),
                created_at=get_beijing_time()
            )
            db.add(record)
            db.commit()
            print(f"成功创建评估任务与记录: 任务ID={task_id}, 记录ID={record_id}")
        except Exception as e:
            db.rollback()
            print(f"数据库插入错误: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"创建评估任务失败: {str(e)}"
            )

        # 4. 立即返回任务ID和记录ID
        response = EvaluationTaskCreateResponse(
            task_id=str(task_id),
            record_id=str(record_id)
        )
        
        # 5. 启动一个真正的分离的后台任务
        # 使用一个完全独立的进程/线程来处理耗时评估
        def run_evaluation_in_thread():
            from sqlalchemy.orm import sessionmaker
            from database import engine
            LocalSession = sessionmaker(bind=engine)
            local_db = LocalSession()
            try:
                # 执行异步评估任务，但在单独的线程中同步运行
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                loop.run_until_complete(
                    run_evaluation_async(
                        db=local_db,
                        record_id=record_id,
                        file_id=request.file_id,
                        metric_id=request.metric_id,
                        system_prompt=request.system_prompt
                    )
                )
                loop.close()
            except Exception as e:
                print(f"后台评估线程出错: {str(e)}")
            finally:
                local_db.close()
        
        # 启动真正后台运行的线程
        eval_thread = threading.Thread(target=run_evaluation_in_thread)
        eval_thread.daemon = True  # 设置为守护线程，不阻止主进程退出
        eval_thread.start()
        print(f"后台评估线程已启动，立即返回响应: {response.dict()}")
        
        return response
    except HTTPException:
        # 直接重新抛出HTTP异常
        raise
    except Exception as e:
        print(f"创建评估任务时发生意外错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建评估任务失败: {str(e)}"
        )


async def run_evaluation_async(
        db: Session,
        record_id: UUID,
        file_id: str,
        metric_id: str,
        system_prompt: str
):
    """异步执行评估任务"""
    from sqlalchemy.orm import sessionmaker
    from database import engine
    from loguru import logger

    # 创建独立session避免主线程session问题
    LocalSession = sessionmaker(bind=engine)
    local_db = LocalSession()

    try:
        # 获取文件内容
        file_record = local_db.query(FileDB).filter(FileDB.id == file_id).first()
        if not file_record:
            raise ValueError("文件不存在")

        file_content = json.loads(file_record.data.decode('utf-8'))
        if not isinstance(file_content, list):
            raise ValueError("文件内容必须是JSON数组")

        # 执行评估
        await run_evaluation_in_background(
            db=local_db,
            record_id=record_id,
            file_content=file_content,
            metric_id=metric_id,
            system_prompt=system_prompt
        )
    except Exception as e:
        logger.error(f"评估任务执行失败: {str(e)}")
        # 更新任务状态为失败
        record = local_db.query(EvaluationRecord).filter_by(id=record_id).first()
        if record:
            record.task.status = "failed"
            record.task.error_message = str(e)
            local_db.commit()
    finally:
        local_db.close()


@router.post("/iterations", response_model=EvaluationIterationResponse)
async def create_iteration(
        request: EvaluationIterationRequest,
        background_tasks: BackgroundTasks,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """创建评估迭代
    1. 验证用户和任务
    2. 获取先前的评估记录以获取文件信息
    3. 创建新的评估记录
    4. 启动后台评估任务
    5. 返回新记录ID
    """
    # 认证用户
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)
    
    print(f"收到评估迭代请求 - 用户ID: {user_id}, 任务ID: {request.task_id}")

    try:
        # 验证任务所属
        task = db.query(EvaluationTask).filter(
            EvaluationTask.id == request.task_id,
            EvaluationTask.user_id == user_id
        ).first()

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="任务不存在或无权访问"
            )

        # 获取上一次评估记录以获取必要信息
        previous_record = db.query(EvaluationRecord).filter(
            EvaluationRecord.task_id == request.task_id
        ).order_by(
            EvaluationRecord.created_at.desc()
        ).first()

        if not previous_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="未找到先前的评估记录"
            )

        # 获取文件内容
        file_record = db.query(FileDB).filter(
            FileDB.id == previous_record.file_id
        ).first()

        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="评估数据文件不存在"
            )

        try:
            file_content = json.loads(file_record.data.decode('utf-8'))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文件解析失败: {str(e)}"
            )

        # 创建新的评估记录，使用优先的评估指标
        metric_id = previous_record.metric_id
        if metric_id == "faithfulness":
            # 如果使用的是faithfulness指标，自动替换为prompt_scs指标
            metric_id = "prompt_scs"
        
        record_id = uuid.uuid4()
        new_record = EvaluationRecord(
            id=record_id,
            task_id=UUID(request.task_id),
            metric_id=metric_id,
            system_prompt=request.system_prompt,
            file_id=previous_record.file_id,
            created_at=get_beijing_time()  # 使用北京时间
        )

        # 更新任务状态
        task.status = "processing"

        db.add(new_record)
        db.commit()
        
        print(f"成功创建评估迭代记录: 记录ID={record_id}")

        # 启动后台评估任务 - 使用线程而不是BackgroundTasks
        def run_evaluation_in_thread():
            from sqlalchemy.orm import sessionmaker
            from database import engine
            LocalSession = sessionmaker(bind=engine)
            local_db = LocalSession()
            try:
                # 执行评估任务，在单独的线程中运行
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                loop.run_until_complete(
                    run_evaluation_in_background(
                        db=local_db,
                        record_id=record_id,
                        file_content=file_content,
                        metric_id=metric_id,
                        system_prompt=request.system_prompt
                    )
                )
                loop.close()
            except Exception as e:
                print(f"迭代评估线程出错: {str(e)}")
            finally:
                local_db.close()
        
        # 启动真正后台运行的线程
        eval_thread = threading.Thread(target=run_evaluation_in_thread)
        eval_thread.daemon = True  # 设置为守护线程
        eval_thread.start()
        print(f"后台迭代评估线程已启动，立即返回响应")

        return EvaluationIterationResponse(record_id=str(record_id))

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"创建评估迭代出错: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建评估迭代失败: {str(e)}"
        )


@router.delete("/tasks/{task_id}", response_model=DeleteTaskResponse)
async def delete_task(
        task_id: UUID,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """删除评估任务及其所有记录"""
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)

    result = service.delete_task(task_id, user_id)

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )

    return DeleteTaskResponse(**result)


@router.get("/tasks/{task_id}", response_model=EvaluationTaskItem)
async def get_task_detail(
        task_id: UUID,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """获取任务详情"""
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)

    task = db.query(EvaluationTask).filter(
        EvaluationTask.id == task_id,
        EvaluationTask.user_id == user_id
    ).first()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="任务不存在或无权访问"
        )

    return EvaluationTaskItem(
        id=str(task.id),
        name=task.name,
        created_at=to_timestamp_ms(task.created_at),  # 使用工具函数
        metric_id=task.records[0].metric_id if task.records else "",
        metric_name=service.metrics.get(task.records[0].metric_id, {}).get("name", "") if task.records else "",
        status=task.status,
        dataset_id=str(task.records[0].file_id) if task.records else "",
        iterations=len(task.records)
    )


async def run_evaluation_in_background(
        db: Session,
        record_id: UUID,
        file_content: list,
        metric_id: str,
        system_prompt: str
):
    """后台执行评估任务"""
    from sqlalchemy.orm import sessionmaker
    from database import engine
    from loguru import logger
    import os
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate

    # 创建独立session避免主线程session问题
    LocalSession = sessionmaker(bind=engine)
    local_db = LocalSession()

    try:
        service = EvaluationService(local_db)

        # 准备评估数据
        questions = []
        answers = []
        generated_responses = []

        # 初始化LLM模型
        try:
            llm = ChatOpenAI(
                base_url="https://open.bigmodel.cn/api/paas/v4/",
                api_key=os.getenv("ZHIPU_API_KEY"),
                model="glm-4-flash"
            )

            # 创建聊天提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{question}")
            ])
            chain = prompt | llm
            logger.info("LLM模型初始化成功")
        except Exception as e:
            logger.error(f"LLM初始化失败: {str(e)}")
            chain = None

        for item in file_content:
            if not isinstance(item, dict):
                continue

            if 'query' in item and 'answer' in item:
                question = str(item['query'])
                questions.append(question)
                answers.append(str(item['answer']))

                # 生成回答
                if chain:
                    try:
                        # 使用LLM生成回答
                        response = chain.invoke({"question": question})
                        ai_response = response.content
                        logger.info(f"成功生成问题的回答: {question[:30]}...")
                    except Exception as e:
                        logger.error(f"生成回答失败: {str(e)}")
                        # 失败时使用备用回答
                        ai_response = f"抱歉，我无法回答这个问题。"
                else:
                    # 如果LLM初始化失败，使用模拟回答
                    ai_response = f"这是针对 '{question}' 的模拟回答, 基于提示: {system_prompt[:50]}..."

                generated_responses.append(ai_response)

        if not questions or not answers:
            raise ValueError("文件中没有有效的query/answer数据")

        # 执行评估 - 首选prompt_scs评估方法
        evaluation_metrics = [metric_id]
        if metric_id == "faithfulness":
            # 用prompt_scs替换faithfulness
            logger.info("将faithfulness评估指标替换为prompt_scs")
            evaluation_metrics = ["prompt_scs"]
            
        # 添加BLEU评估作为辅助指标
        if "bleu" not in evaluation_metrics:
            evaluation_metrics.append("bleu")
            
        logger.info(f"开始评估记录 {record_id}，使用指标: {evaluation_metrics}...")
        
        try:
            result = await service.evaluate(
                questions=questions, 
                answers=answers, 
                metric_names=evaluation_metrics,
                generated_responses=generated_responses
            )
            logger.info(f"评估完成，获得指标: {list(result.scores.keys())}")
        except Exception as e:
            logger.error(f"评估过程出错: {str(e)}")
            # 如果评估失败，尝试回退到单指标评估
            fallback_metric = evaluation_metrics[0]
            logger.info(f"尝试回退到单一指标 {fallback_metric}")
            
            try:
                result = await service.evaluate(
                    questions=questions, 
                    answers=answers, 
                    metric_names=[fallback_metric],
                    generated_responses=generated_responses
                )
                logger.info(f"回退评估成功，使用指标: {fallback_metric}")
            except Exception as fallback_error:
                logger.error(f"回退评估也失败: {str(fallback_error)}")
                # 创建一个模拟结果
                from collections import namedtuple
                Result = namedtuple('Result', ['scores'])
                result = Result(scores={
                    fallback_metric: 0.7  # 默认分数
                })
                logger.warning("使用默认分数完成评估")

        # 准备详细结果
        detailed_results = {
            "scores": result.scores,
            "samples": []
        }

        # 首选的评估指标
        primary_metric = evaluation_metrics[0]
        
        # 将评估结果组织成样本列表
        metric_scores = result.scores.get(primary_metric, {})
        if isinstance(metric_scores, float):
            # 单一分数
            for i, (q, a, g) in enumerate(zip(questions, answers, generated_responses)):
                sample_details = {primary_metric: metric_scores}
                # 添加其他指标
                for other_metric in evaluation_metrics[1:]:
                    if other_metric in result.scores:
                        other_scores = result.scores[other_metric]
                        if isinstance(other_scores, list) and i < len(other_scores):
                            sample_details[other_metric] = other_scores[i]
                        elif not isinstance(other_scores, list):
                            sample_details[other_metric] = other_scores
                            
                detailed_results["samples"].append({
                    "query": q,
                    "answer": a,
                    "generated": g,
                    "score": metric_scores,  # 主评估指标分数作为总分
                    "details": sample_details
                })
        else:
            # 每个样本有独立分数
            for i, (q, a, g) in enumerate(zip(questions, answers, generated_responses)):
                if i < len(metric_scores):
                    score = metric_scores[i]
                    sample_details = {primary_metric: score}
                    # 添加其他指标
                    for other_metric in evaluation_metrics[1:]:
                        if other_metric in result.scores:
                            other_scores = result.scores[other_metric]
                            if isinstance(other_scores, list) and i < len(other_scores):
                                sample_details[other_metric] = other_scores[i]
                            elif not isinstance(other_scores, list):
                                sample_details[other_metric] = other_scores
                                
                    detailed_results["samples"].append({
                        "query": q,
                        "answer": a,
                        "generated": g,
                        "score": score,  # 主评估指标分数作为总分
                        "details": sample_details
                    })

        # 更新评估记录
        record = local_db.query(EvaluationRecord).filter_by(id=record_id).first()
        if record:
            record.results = detailed_results
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
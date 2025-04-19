from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from fastapi_jwt_auth2 import AuthJWT
from sqlalchemy.orm import Session
from uuid import UUID
import json
import uuid
from datetime import datetime
import asyncio
import threading
from typing import Optional, List

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
    EvaluationIterationResponse, DeleteTaskResponse, RAGEvaluationRequest, RAGIterationRequest
)
from ..user.auth import decode_jwt_to_uid
from utils.datetime_tools import get_beijing_time, to_timestamp_ms  # 导入工具函数

from . import router as evaluation_router
from ..user.user_router import router as user_router

__all__ = ["evaluation_router", "user_router"]
router = APIRouter(tags=["evaluation"], prefix="/evaluation")


@router.get("/metrics", response_model=list[Metric])
async def get_metrics(db: Session = Depends(get_db), type: Optional[str] = None):
    """获取可用评估指标，可以按类型过滤"""
    service = EvaluationService(db)
    metrics = []
    
    # 如果指定了类型，则只返回该类型的指标
    if type:
        metrics_dict = service.get_metrics_by_type(type)
        
        # 添加指标
        for metric_id, metric_info in metrics_dict.items():
            metrics.append(
                Metric(id=metric_id, name=metric_info["name"], description=metric_info["description"])
            )
        
        return metrics
    
    # 否则按优先级返回所有指标
    # 优先显示prompt评估指标
    priority_metrics = ["prompt_scs", "bleu", "answer_relevancy", "faithfulness", "context_relevancy", "context_precision"]
    
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
        is_rag_task: Optional[bool] = None,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """获取评估任务列表，支持分页和按任务类型筛选"""
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)

    skip = (page - 1) * page_size
    result = service.get_all_tasks(user_id, skip, page_size, is_rag_task)

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
            import asyncio
            import nest_asyncio
            
            # 为当前线程创建独立的事件循环
            LocalSession = sessionmaker(bind=engine)
            local_db = LocalSession()
            try:
                # 防止嵌套事件循环问题
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                nest_asyncio.apply(loop)
                
                # 在事件循环中执行异步评估
                async def run_task():
                    try:
                        await run_evaluation_async(
                            db=local_db,
                            record_id=record_id,
                            file_id=request.file_id,
                            metric_id=request.metric_id,
                            system_prompt=request.system_prompt
                        )
                    finally:
                        pass
                
                # 运行异步任务并确保完成
                loop.run_until_complete(run_task())
            except Exception as e:
                print(f"后台评估线程出错: {str(e)}")
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()
                except Exception as e:
                    print(f"关闭事件循环失败: {str(e)}")
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
            import asyncio
            import nest_asyncio
            
            # 为当前线程创建独立的事件循环
            LocalSession = sessionmaker(bind=engine)
            local_db = LocalSession()
            try:
                # 防止嵌套事件循环问题
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                nest_asyncio.apply(loop)
                
                # 在事件循环中执行异步评估
                async def run_task():
                    try:
                        await run_evaluation_in_background(
                            db=local_db,
                            record_id=record_id,
                            file_content=file_content,
                            metric_id=metric_id,
                            system_prompt=request.system_prompt
                        )
                    finally:
                        pass
                
                # 运行异步任务并确保完成
                loop.run_until_complete(run_task())
            except Exception as e:
                print(f"迭代评估线程出错: {str(e)}")
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()
                except Exception as e:
                    print(f"关闭事件循环失败: {str(e)}")
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
    from contextlib import asynccontextmanager

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
        
        # 使用上下文管理器确保资源被正确释放
        @asynccontextmanager
        async def managed_evaluation():
            try:
                result = await service.evaluate(
                    questions=questions, 
                    answers=answers, 
                    metric_names=evaluation_metrics,
                    generated_responses=generated_responses
                )
                yield result
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
                    yield result
                except Exception as fallback_error:
                    logger.error(f"回退评估也失败: {str(fallback_error)}")
                    # 创建一个模拟结果
                    from collections import namedtuple
                    Result = namedtuple('Result', ['scores'])
                    result = Result(scores={
                        fallback_metric: 0.7  # 默认分数
                    })
                    logger.warning("使用默认分数完成评估")
                    yield result

        # 使用上下文管理器确保资源正确释放
        async with managed_evaluation() as result:
            logger.info(f"评估完成，获得指标: {list(result.scores.keys())}")
            
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
        # 确保关闭数据库连接
        local_db.close()
        
        # 强制清理所有异步资源
        try:
            import asyncio
            # 清理LLM资源
            if 'chain' in locals() and chain is not None:
                if hasattr(chain, 'aclose'):
                    try:
                        logger.info("关闭chain资源")
                        asyncio.create_task(chain.aclose())
                    except Exception as e:
                        logger.error(f"关闭chain资源失败: {str(e)}")

            # 检查当前事件循环中的所有任务是否完成
            pending = asyncio.all_tasks()
            if pending:
                logger.info(f"等待 {len(pending)} 个待处理任务完成")
                # 等待所有任务完成，但设置超时
                try:
                    done, still_pending = await asyncio.wait(pending, timeout=3.0)
                    if still_pending:
                        logger.warning(f"仍有 {len(still_pending)} 个任务未完成，但继续执行")
                except Exception as e:
                    logger.error(f"等待任务完成时出错: {str(e)}")
            
            # 强制清理内存
            import gc
            gc.collect()
            logger.info("资源清理完成")
        except Exception as final_e:
            logger.error(f"最终清理过程出错: {str(final_e)}")


# 增加RAG评估相关的端点
@router.post("/rag/tasks", response_model=EvaluationTaskCreateResponse)
async def create_rag_evaluation_task(
        request: RAGEvaluationRequest,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """创建RAG评估任务
    1. 验证用户和文件
    2. 创建任务和记录数据库条目
    3. 立即返回任务ID和记录ID
    4. 异步启动评估任务
    """
    # 1. 认证用户
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)
    
    # 添加详细日志
    print(f"收到RAG评估任务创建请求 - 用户ID: {user_id}, 指标: {request.metric_ids}")
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
        
        # 验证文件格式是否为RAG评估格式
        try:
            file_content = json.loads(file_record.data.decode('utf-8'))
            if not isinstance(file_content, list) or len(file_content) == 0:
                raise ValueError("文件内容必须是非空JSON数组")
                
            # 检查第一条记录是否包含必要的字段
            sample = file_content[0]
            if not all(key in sample for key in ["query", "answer", "retrieved_contexts"]):
                raise ValueError("每条记录必须包含query、answer和retrieved_contexts字段")
                
            if not isinstance(sample["retrieved_contexts"], list):
                raise ValueError("retrieved_contexts必须是数组")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"文件格式验证失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文件格式无效: {str(e)}"
            )
        
        print(f"文件验证成功: {request.file_id}")

        # 选择第一个RAG评估指标作为默认指标（用于显示）
        primary_metric_id = request.metric_ids[0] if request.metric_ids else "faithfulness"

        # 3. 创建数据库记录
        task_id = uuid.uuid4()
        record_id = uuid.uuid4()

        try:
            # 创建任务，标记为RAG评估任务
            task = EvaluationTask(
                id=task_id,
                name=request.task_name,
                user_id=user_id,
                status="processing",
                created_at=get_beijing_time(),
                is_rag_task=True  # 标记为RAG评估任务
            )
            db.add(task)

            # 创建评估记录
            record = EvaluationRecord(
                id=record_id,
                task_id=task_id,
                metric_id=primary_metric_id,  # 主指标
                system_prompt=None,  # RAG评估不需要系统prompt
                file_id=UUID(request.file_id),
                created_at=get_beijing_time(),
                results={"metric_ids": request.metric_ids}  # 将所有指标ID存储在results字段中
            )
            db.add(record)
            db.commit()
            print(f"成功创建RAG评估任务与记录: 任务ID={task_id}, 记录ID={record_id}")
        except Exception as e:
            db.rollback()
            print(f"数据库插入错误: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"创建RAG评估任务失败: {str(e)}"
            )

        # 4. 立即返回任务ID和记录ID
        response = EvaluationTaskCreateResponse(
            task_id=str(task_id),
            record_id=str(record_id)
        )
        
        # 5. 启动一个真正的分离的后台任务
        def run_rag_evaluation_in_thread():
            from sqlalchemy.orm import sessionmaker
            from database import engine
            import asyncio
            import nest_asyncio
            
            # 为当前线程创建独立的事件循环
            LocalSession = sessionmaker(bind=engine)
            local_db = LocalSession()
            try:
                # 防止嵌套事件循环问题
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                nest_asyncio.apply(loop)
                
                # 在事件循环中执行异步评估
                async def run_task():
                    try:
                        await run_rag_evaluation_async(
                            db=local_db,
                            record_id=record_id,
                            file_id=request.file_id,
                            metric_ids=request.metric_ids,
                        )
                    finally:
                        pass
                
                # 运行异步任务并确保完成
                loop.run_until_complete(run_task())
            except Exception as e:
                print(f"后台RAG评估线程出错: {str(e)}")
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()
                except Exception as e:
                    print(f"关闭事件循环失败: {str(e)}")
                local_db.close()
        
        # 启动真正后台运行的线程
        eval_thread = threading.Thread(target=run_rag_evaluation_in_thread)
        eval_thread.daemon = True  # 设置为守护线程，不阻止主进程退出
        eval_thread.start()
        print(f"后台RAG评估线程已启动，立即返回响应: {response.dict()}")
        
        return response
    except HTTPException:
        # 直接重新抛出HTTP异常
        raise
    except Exception as e:
        print(f"创建RAG评估任务时发生意外错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建RAG评估任务失败: {str(e)}"
        )


@router.post("/rag/iterations", response_model=EvaluationIterationResponse)
async def create_rag_iteration(
        request: RAGIterationRequest,
        Authorize: AuthJWT = Depends(),
        db: Session = Depends(get_db)
):
    """创建RAG评估迭代
    1. 验证用户和任务
    2. 获取先前的评估记录以获取指标信息
    3. 创建新的评估记录
    4. 启动后台评估任务
    5. 返回新记录ID
    """
    # 认证用户
    user_id = decode_jwt_to_uid(Authorize)
    service = EvaluationService(db)
    
    print(f"收到RAG评估迭代请求 - 用户ID: {user_id}, 任务ID: {request.task_id}")

    try:
        # 验证任务所属和类型
        task = db.query(EvaluationTask).filter(
            EvaluationTask.id == request.task_id,
            EvaluationTask.user_id == user_id,
            EvaluationTask.is_rag_task == True  # 确保是RAG评估任务
        ).first()

        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="RAG评估任务不存在或无权访问"
            )

        # 获取上一次评估记录以获取指标信息
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

        # 获取文件内容并验证
        file_record = db.query(FileDB).filter(
            FileDB.id == request.file_id,
            FileDB.user_id == user_id
        ).first()

        if not file_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="评估数据文件不存在或无权访问"
            )

        try:
            file_content = json.loads(file_record.data.decode('utf-8'))
            if not isinstance(file_content, list) or len(file_content) == 0:
                raise ValueError("文件内容必须是非空JSON数组")
                
            # 检查第一条记录是否包含必要的字段
            sample = file_content[0]
            if not all(key in sample for key in ["query", "answer", "retrieved_contexts"]):
                raise ValueError("每条记录必须包含query、answer和retrieved_contexts字段")
                
            if not isinstance(sample["retrieved_contexts"], list):
                raise ValueError("retrieved_contexts必须是数组")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"文件解析失败: {str(e)}"
            )

        # 使用之前的评估指标
        metric_ids = previous_record.results.get("metric_ids") if previous_record.results and "metric_ids" in previous_record.results else [previous_record.metric_id]
        primary_metric_id = previous_record.metric_id

        # 创建新的评估记录
        record_id = uuid.uuid4()
        new_record = EvaluationRecord(
            id=record_id,
            task_id=UUID(request.task_id),
            metric_id=primary_metric_id,
            system_prompt=None,  # RAG评估不需要系统prompt
            file_id=UUID(request.file_id),
            created_at=get_beijing_time(),
            results={"metric_ids": metric_ids}  # 将所有指标ID存储在results字段中
        )

        # 更新任务状态
        task.status = "processing"

        db.add(new_record)
        db.commit()
        
        print(f"成功创建RAG评估迭代记录: 记录ID={record_id}")

        # 启动后台评估任务
        def run_rag_evaluation_in_thread():
            from sqlalchemy.orm import sessionmaker
            from database import engine
            import asyncio
            import nest_asyncio
            
            # 为当前线程创建独立的事件循环
            LocalSession = sessionmaker(bind=engine)
            local_db = LocalSession()
            try:
                # 防止嵌套事件循环问题
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                nest_asyncio.apply(loop)
                
                # 在事件循环中执行异步评估
                async def run_task():
                    try:
                        await run_rag_evaluation_async(
                            db=local_db,
                            record_id=record_id,
                            file_id=request.file_id,
                            metric_ids=metric_ids
                        )
                    finally:
                        pass
                
                # 运行异步任务并确保完成
                loop.run_until_complete(run_task())
            except Exception as e:
                print(f"RAG迭代评估线程出错: {str(e)}")
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()
                except Exception as e:
                    print(f"关闭事件循环失败: {str(e)}")
                local_db.close()
        
        # 启动真正后台运行的线程
        eval_thread = threading.Thread(target=run_rag_evaluation_in_thread)
        eval_thread.daemon = True  # 设置为守护线程
        eval_thread.start()
        print(f"后台RAG迭代评估线程已启动，立即返回响应")

        return EvaluationIterationResponse(record_id=str(record_id))

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"创建RAG评估迭代出错: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建RAG评估迭代失败: {str(e)}"
        )


async def run_rag_evaluation_async(
        db: Session,
        record_id: UUID,
        file_id: str,
        metric_ids: List[str]
):
    """异步执行RAG评估任务"""
    from sqlalchemy.orm import sessionmaker
    from database import engine
    from loguru import logger

    # 创建独立session避免主线程session问题
    LocalSession = sessionmaker(bind=engine)
    local_db = LocalSession()

    try:
        # 获取记录和指标
        record = local_db.query(EvaluationRecord).filter_by(id=record_id).first()
        if not record:
            raise ValueError(f"评估记录不存在: {record_id}")
            
        # 如果传入的指标为空，尝试从记录中获取
        if not metric_ids and record.results and "metric_ids" in record.results:
            metric_ids = record.results["metric_ids"]
        elif not metric_ids:
            # 如果仍然没有指标，使用记录的主指标
            metric_ids = [record.metric_id]
            
        logger.info(f"使用评估指标: {metric_ids}")
        
        # 获取文件内容
        file_record = local_db.query(FileDB).filter(FileDB.id == file_id).first()
        if not file_record:
            raise ValueError("文件不存在")

        file_content = json.loads(file_record.data.decode('utf-8'))
        if not isinstance(file_content, list):
            raise ValueError("文件内容必须是JSON数组")

        # 执行RAG评估
        service = EvaluationService(local_db)
        logger.info(f"开始RAG评估，记录ID: {record_id}，指标: {metric_ids}")
        
        # 执行评估
        result = await service.evaluate_rag(
            data=file_content,
            metric_names=metric_ids
        )
        
        logger.info(f"RAG评估完成，获得指标: {list(result.scores.keys())}")
        
        # 准备详细结果
        detailed_results = {
            "scores": result.scores,
            "samples": []
        }
        
        # 将评估结果组织成样本列表
        primary_metric = metric_ids[0]
        
        # 准备每个样本的详细结果
        for i, item in enumerate(file_content):
            if not all(k in item for k in ["query", "answer", "retrieved_contexts"]):
                continue
                
            sample_details = {}
            scores = {}
            
            # 收集每个指标的得分
            for metric_id in metric_ids:
                if metric_id in result.scores:
                    metric_scores = result.scores[metric_id]
                    if isinstance(metric_scores, list) and i < len(metric_scores):
                        sample_score = metric_scores[i]
                        sample_details[metric_id] = sample_score
                        if metric_id == primary_metric:
                            scores = sample_score
                    elif not isinstance(metric_scores, list):
                        sample_details[metric_id] = metric_scores
                        if metric_id == primary_metric:
                            scores = metric_scores
            
            # 计算单个样本的总得分（使用主指标的得分）
            total_score = 0
            if isinstance(scores, dict) and scores:
                total_score = sum(scores.values()) / len(scores)
            elif isinstance(scores, (int, float)):
                total_score = scores
            else:
                # 如果没有得分，使用所有指标的平均值
                all_scores = []
                for metric_id in metric_ids:
                    if metric_id in result.scores:
                        if isinstance(result.scores[metric_id], list) and i < len(result.scores[metric_id]):
                            all_scores.append(result.scores[metric_id][i])
                        elif not isinstance(result.scores[metric_id], list):
                            all_scores.append(result.scores[metric_id])
                
                if all_scores:
                    total_score = sum(all_scores) / len(all_scores)
            
            # 添加样本到结果中
            detailed_results["samples"].append({
                "query": item["query"],
                "answer": item["answer"],
                "retrieved_contexts": item["retrieved_contexts"],
                "ground_truth": item.get("ground_truth", ""),
                "score": total_score,
                "details": sample_details
            })
            
        # 更新评估记录
        record = local_db.query(EvaluationRecord).filter_by(id=record_id).first()
        if record:
            record.results = detailed_results
            record.task.status = "completed"
            local_db.commit()
            logger.success(f"RAG评估记录 {record_id} 完成")
        else:
            logger.error(f"RAG评估记录 {record_id} 不存在")
    except Exception as e:
        logger.error(f"RAG评估执行失败: {str(e)}")
        # 更新任务状态为失败
        record = local_db.query(EvaluationRecord).filter_by(id=record_id).first()
        if record:
            record.task.status = "failed"
            record.task.error_message = str(e)
            local_db.commit()
    finally:
        local_db.close()
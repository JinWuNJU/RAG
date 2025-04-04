from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import text, exc

from ..database import get_db
from .service import *
from model.knowledge_base.knowledge_base_model import *
from model.knowledge_base.schemas import *


# 创建知识库相关的API路由，设置标签和前缀
router = APIRouter(tags=["Knowledge Bases"], prefix="/knowledge_bases")

@router.get("/test")
def test_endpoint():
    return {"message": "OK1"}

@router.post("/", response_model=KnowledgeBaseCreateResponse)
async def create_knowledge_base(
        data: KnowledgeBaseCreate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    """
    创建知识库（异步处理文件分块）

    参数:
    - data: 包含知识库创建所需数据的请求体，包括：
        - name: 知识库名称
        - description: 知识库描述
        - file_ids: 关联的文件ID列表
        - chunk_size: 文档分块大小
        - overlap_size: 分块重叠大小
        - hybrid_ratio: 混合检索比例（预留参数）

    功能:
    1. 创建知识库元数据记录，初始状态为"building"
    2. 在后台异步处理文件分块：
       - 使用中文分词库
       - 根据chunk_size和overlap_size分割文件内容
       - 将分块存入knowledge_base_chunks表
    3. 向量列(embedding)暂不填充

    返回:
    - 创建的知识库基本信息，包括knowledge_base_id和状态
    """
    # 创建知识库元数据
    try:
        # 创建知识库元数据
        kb = KnowledgeBase(
            name=data.name,
            chunk_size=data.chunk_size,
            overlap_size=data.overlap_size,
            description=data.description,
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)

        # 提交后台处理任务
        processor = TextFileProcessor(db)
        background_tasks.add_task(
            processor.process_files,
            kb_id=kb.id,
            file_ids=data.file_ids
        )

        return {
            "knowledge_base_id": kb.id,
            "status": "building"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail=str(e))


@router.get("/",response_model=List[KnowledgeBaseListItem])
async def list_knowledge_bases(
        db: Session = Depends(get_db),
        skip: int = 0,
        limit: int = 100
) -> List[KnowledgeBaseListItem]:
    """
    获取知识库列表API

    参数:
    - skip: 跳过的记录数（用于分页），默认为0
    - limit: 每页返回的最大记录数，默认为100，最大1000

    返回:
    - List[KnowledgeBaseListItem]: 知识库基本信息列表，包含:
      - knowledge_base_id: 知识库唯一标识
      - name: 知识库名称
      - description: 知识库描述（可选）
      - created_at: 创建时间(ISO 8601格式)
      - status: 知识库状态（building/completed）


    错误码:
    - 500: 服务器内部错误
    """
    try:
        # 从数据库查询知识库列表（按创建时间倒序）
        knowledge_bases = db.query(KnowledgeBase) \
            .order_by(KnowledgeBase.created_at.desc()) \
            .offset(skip) \
            .limit(limit) \
            .all()

        # 转换为响应模型
        return [
            KnowledgeBaseListItem(
                knowledge_base_id=kb.id,
                name=kb.name,
                description=kb.description,
                created_at=kb.created_at,
                status=kb.status
            )
            for kb in knowledge_bases
        ]

    except Exception as e:
        logger.error(f"获取知识库列表失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/{knowledge_base_id}",response_model=KnowledgeBaseDetailResponse)
async def get_knowledge_base_detail(
    knowledge_base_id: UUID,
    db: Session = Depends(get_db)
) -> KnowledgeBaseDetailResponse:
    """
    获取知识库详情API

    参数:
    - knowledge_base_id: 知识库唯一标识(UUID格式)

    返回:
    - KnowledgeBaseDetailResponse: 知识库详细信息，包含:
      - knowledge_base_id: 知识库唯一标识
      - name: 知识库名称
      - description: 知识库描述（可选）
      - created_at: 创建时间(ISO 8601格式)
      - status: 知识库状态（building/completed/partial_completed）
      - chunk_size: 文档分块大小
      - overlap_size: 分块重叠大小
      - hybrid_ratio: 混合检索比例


    错误码:
    - 404: 知识库不存在
    - 500: 服务器内部错误
    """
    try:
        # 查询知识库详情
        kb = db.query(KnowledgeBase).filter_by(id=knowledge_base_id).first()
        if not kb:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base {knowledge_base_id} not found"
            )

        return KnowledgeBaseDetailResponse(
            knowledge_base_id=kb.id,
            name=kb.name,
            description=kb.description,
            created_at=kb.created_at,
            status=kb.status,
            chunk_size=kb.chunk_size,
            overlap_size=kb.overlap_size,
            hybrid_ratio=kb.hybrid_ratio
        )

    except HTTPException:
        raise  # 直接抛出已知的HTTP异常
    except Exception as e:
        logger.error(f"获取知识库详情失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@router.post("/{knowledge_base_id}/search",response_model=List[SearchResult])
async def search_knowledge_base(
        knowledge_base_id: UUID,
        request: SearchRequest,
        db: Session = Depends(get_db)
) -> List[SearchResult]:
    """
    搜索知识库内容API

    参数:
    - knowledge_base_id: 知识库唯一标识
    - request: 搜索请求体，包含:
      - query: 搜索关键词
      - limit: 返回结果数量限制（默认10，最大100）

    返回:
    - List[SearchResult]: 匹配的文档片段列表



    错误码:
    - 400: 搜索关键词为空
    - 404: 知识库不存在或无匹配结果
    - 500: 服务器内部错误
    """
    # 参数验证
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="搜索关键词不能为空")

    # 限制最大返回数量
    limit = min(request.limit, 100)

    try:
        # 执行PGroonga全文检索
        results = db.query(
            KnowledgeBaseChunk.content,
            KnowledgeBaseChunk.file_name,
            KnowledgeBaseChunk.file_id,
            KnowledgeBaseChunk.chunk_index
        ).filter(
            KnowledgeBaseChunk.knowledge_base_id == knowledge_base_id,
            text(f"knowledge_base_chunks.content &@~ '{request.query}'")
        ).order_by(
            text("pgroonga_score(knowledge_base_chunks.tableoid, knowledge_base_chunks.ctid) DESC")
        ).limit(limit).all()

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"知识库 {knowledge_base_id} 中未找到匹配内容"
            )

        return [
            SearchResult(
                content=result.content,
                file_name=result.file_name,
                file_id=result.file_id,
                chunk_index=result.chunk_index
            )
            for result in results
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="服务器内部错误，请稍后再试"
        )

# 索引健康检查端点（可选）
@router.get("/search/status")
async def check_search_status(db: Session = Depends(get_db)):
    """检查搜索引擎状态"""
    try:
        # 检查PGroonga索引
        db.execute(text("SELECT pgroonga_command('status')")).fetchone()
        # 检查向量索引
        db.execute(text("SELECT ivfflat_probes('ivfflat_embedding_index')"))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(503, detail=f"搜索引擎异常: {str(e)}")


@router.put("/{knowledge_base_id}/rebuild", response_model=KnowledgeBaseCreateResponse)
async def rebuild_knowledge_base(
        knowledge_base_id: UUID,
        request: KnowledgeBaseCreate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    """
    重建知识库（异步处理文件分块）

    参数:
    - request: 包含知识库创建所需数据的请求体，包括：
        - name: 知识库名称
        - description: 知识库描述
        - file_ids: 关联的文件ID列表
        - chunk_size: 文档分块大小
        - overlap_size: 分块重叠大小
        - hybrid_ratio: 混合检索比例（预留参数）

    功能:
    1. 更新知识库元数据记录，更改状态为"building"
    2. 删除原有的分块记录
    3. 在后台异步处理文件分块：
       - 使用中文分词库
       - 根据chunk_size和overlap_size分割文件内容
       - 将分块存入knowledge_base_chunks表
    4. 向量列(embedding)暂不填充

    返回:
    - 创建的知识库基本信息，包括knowledge_base_id和状态
    """
    try:
        # 开启事务
        db.begin()

        # 检查知识库是否存在
        kb = db.query(KnowledgeBase).filter_by(id=knowledge_base_id).first()
        if not kb:
            db.rollback()
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base {knowledge_base_id} not found"
            )

        # # 获取关联文件列表（用于后续重建）
        # files = db.query(FileDB) \
        #     .filter_by(knowledge_base_id=knowledge_base_id) \
        #     .all()
        # if not files:
        #     db.rollback()
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail="No files associated with this knowledge base"
        #     )

        # 删除旧的分块数据（批量删除提高性能）
        deleted_count = db.query(KnowledgeBaseChunk) \
            .filter_by(knowledge_base_id=knowledge_base_id) \
            .delete(synchronize_session=False)
        logger.info(f"Deleted {deleted_count} old chunks for KB {knowledge_base_id}")

        # 更新知识库参数
        kb.chunk_size = request.chunk_size
        kb.overlap_size = request.overlap_size
        kb.hybrid_ratio = request.hybrid_ratio
        kb.status = "building"

        db.commit()

        # 启动异步重建任务
        try:

            # 提交后台处理任务
            processor = TextFileProcessor(db)

            background_tasks.add_task(
                processor.process_files,
                kb_id=knowledge_base_id,
                file_ids=request.file_ids
            )
            logger.info(f"Started rebuild task for KB {knowledge_base_id}")
        except Exception as e:
            logger.error(f"Failed to start rebuild task: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to start rebuild process"
            )

        return KnowledgeBaseCreateResponse(
            knowledge_base_id=knowledge_base_id,
            status=kb.status
        )

    except ValueError as e:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except exc.SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during rebuild: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Database operation failed"
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error during rebuild: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.delete(
    "/{knowledge_base_id}",
    status_code=204,
    summary="删除知识库",
    description="删除知识库元数据及关联数据表记录（不处理实际文件删除）",
    responses={
        204: {"description": "成功删除"},
        404: {"description": "知识库不存在"},
        500: {"description": "数据库操作失败"}
    }
)
async def delete_knowledge_base(
        knowledge_base_id: UUID,
        db: Session = Depends(get_db)
):
    """
    删除知识库API

    安全措施：
    1. 使用数据库事务保证数据一致性
    2. 级联删除关联的分块记录
    3. 严格验证知识库存在性

    注意：
    - 不会删除文件系统中的原始文件
    - 删除操作不可逆
    """
    try:
        # 开启事务
        db.begin()

        # 检查知识库是否存在
        kb = db.query(KnowledgeBase).filter_by(id=knowledge_base_id).first()
        if not kb:
            db.rollback()
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base {knowledge_base_id} not found"
            )

        # 级联删除分块数据（使用批量删除提高性能）
        db.query(KnowledgeBaseChunk) \
            .filter_by(knowledge_base_id=knowledge_base_id) \
            .delete(synchronize_session=False)

        # 删除主记录
        db.delete(kb)
        db.commit()

        # 记录审计日志（示例）
        logger.info(f"Deleted knowledge base: {knowledge_base_id}")

        return {"detail": "Knowledge base deleted successfully"}

    except exc.SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error deleting KB {knowledge_base_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Database operation failed"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error deleting KB {knowledge_base_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
from typing import Tuple

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Response, Query
from fastapi_jwt_auth2 import AuthJWT
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import text, exc, func

from database import get_db
from database.model.knowledge_base import *
from database.model.file import FileDB
from rest_model.knowledge_base import *
from .service import *
from ..user import auth
import urllib.parse

# 创建知识库相关的API路由，设置标签和前缀
router = APIRouter(tags=["Knowledge Bases"], prefix="/knowledge_bases")

@router.get("/test")
def test_endpoint():
    return {"message": "OK1"}

@router.post("/", response_model=KnowledgeBaseCreateResponse)
async def create_knowledge_base(
        data: KnowledgeBaseCreate,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
        Authorize: AuthJWT = Depends()
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
    user_id = auth.decode_jwt_to_uid(Authorize)
    # 创建知识库元数据
    try:
        # 创建知识库元数据
        kb = KnowledgeBase(
            name=data.name,
            chunk_size=data.chunk_size,
            overlap_size=data.overlap_size,
            description=data.description,
            uploader_id=user_id,
            is_public=data.is_public,
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


@router.get("/", response_model=PaginatedResponse)
async def list_knowledge_bases(
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends(),
    name: Optional[str] = Query(None, min_length=1),  # 新增关键词参数
    page: int = Query(0, ge=0),                       # 分页参数标准化
    limit: int = Query(10, ge=1, le=1000)            # 限制最大1000
):
    user_id = auth.decode_jwt_to_uid(Authorize)
    try:
        # 基础查询条件
        query = db.query(KnowledgeBase).filter(
            (KnowledgeBase.is_public == True) |
            (KnowledgeBase.uploader_id == user_id)
        )

        if name:
            query = query.filter(
                KnowledgeBase.name.op('&@~')(name)  # 直接使用列操作符
            )

        # 分页计算
        total = query.count()
        total_pages = (total + limit - 1) // limit  # 向上取整

        # 执行分页查询
        knowledge_bases = query.order_by(KnowledgeBase.created_at.desc())\
                               .offset(page * limit)\
                               .limit(limit)\
                               .all()

        return PaginatedResponse(
            items=[
                KnowledgeBaseListItem(
                    knowledge_base_id=kb.id,
                    name=kb.name,
                    description=kb.description,
                    created_at=kb.created_at,
                    status=kb.status,
                    uploader_id=kb.uploader_id,
                    is_public=kb.is_public
                ) for kb in knowledge_bases
            ],
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages
        )

    except Exception as e:
        logger.error(f"获取知识库列表失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

def query_knowledge_base(db: Session, user_id: uuid.UUID, knowledge_base_id: UUID) -> KnowledgeBase:
    """查询知识库"""
    kb = db.query(KnowledgeBase).filter(
        (KnowledgeBase.id == knowledge_base_id) &
        ((KnowledgeBase.is_public == True) | (KnowledgeBase.uploader_id == user_id))
    ).first()

    if not kb:
        raise HTTPException(
            status_code=404,
            detail=f"知识库{knowledge_base_id}无法访问或不存在"
        )
    return kb

@router.get("/{knowledge_base_id}",response_model=KnowledgeBaseDetailResponse)
async def get_knowledge_base_detail(
    knowledge_base_id: UUID,
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends()
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
      - files: 知识库关联的文件列表，包含:
        - file_id: 文件ID
        - file_name: 文件名称
        - file_size: 文件大小
        - chunk_count: 分块数量
        - created_at: 文件创建时间


    错误码:
    - 404: 知识库不存在
    - 500: 服务器内部错误
    """
    user_id = auth.decode_jwt_to_uid(Authorize)
    try:
        kb = query_knowledge_base(db, user_id, knowledge_base_id)
        
        # 查询知识库所关联的文件信息
        file_info_query = (
            db.query(
                KnowledgeBaseChunk.file_id,
                KnowledgeBaseChunk.file_name,
                func.count(KnowledgeBaseChunk.id).label('chunk_count')
            )
            .filter(KnowledgeBaseChunk.knowledge_base_id == knowledge_base_id)
            .group_by(KnowledgeBaseChunk.file_id, KnowledgeBaseChunk.file_name)
            .all()
        )
        
        # 构建文件列表
        files = []
        for file_id, file_name, chunk_count in file_info_query:
            # 获取文件大小和创建时间
            file_db = db.query(FileDB).filter(FileDB.id == file_id).first()
            if file_db:
                files.append(
                    KnowledgeBaseFile(
                        file_id=file_id,
                        file_name=file_name,
                        file_size=file_db.size,
                        chunk_count=chunk_count,
                        created_at=file_db.created_at
                    )
                )

        return KnowledgeBaseDetailResponse(
            knowledge_base_id=kb.id,
            name=kb.name,
            description=kb.description,
            created_at=kb.created_at,
            status=kb.status,
            chunk_size=kb.chunk_size,
            overlap_size=kb.overlap_size,
            hybrid_ratio=kb.hybrid_ratio,
            uploader_id=kb.uploader_id,
            is_public=kb.is_public,
            files=files
        )

    except HTTPException:
        raise  # 直接抛出已知的HTTP异常
    except Exception as e:
        logger.error(f"获取知识库详情失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


def get_text_search_results(db: Session, knowledge_base_id: uuid.UUID, query: str, limit: int) -> List[Tuple[KnowledgeBaseChunk, float]]:
    """使用pgroonga进行全文检索"""
    score_expr = text("pgroonga_score(tableoid, ctid) as score")
    query_expr = text("content &@~ :query")

    results = db.query(
        KnowledgeBaseChunk,
        score_expr
    ).filter(
        KnowledgeBaseChunk.knowledge_base_id == knowledge_base_id,
        query_expr
    ).params(query=query).order_by(
        text("score DESC")
    ).limit(limit).all()

    if not results:
        return []

    return [(chunk, score) for chunk, score in results]


async def get_vector_search_results(db: Session, knowledge_base_id: uuid.UUID, query: str, limit: int) -> List[Tuple[KnowledgeBaseChunk, float]]:
    """使用向量相似度搜索"""
    # 获取查询的嵌入向量
    query_embedding = await EmbeddingService.get_instance().embed_text(query)
    if query_embedding is None:
        return []

    # 转换为适合pgvector的格式
    query_embedding_pg = query_embedding.tolist()

    # 使用pgvector的内置余弦相似度操作符
    chunks = db.query(
        KnowledgeBaseChunk,
        (1 - KnowledgeBaseChunk.embedding.cosine_distance(query_embedding_pg)).label("score")
    ).filter(
        KnowledgeBaseChunk.knowledge_base_id == knowledge_base_id,
        KnowledgeBaseChunk.embedding.isnot(None)
    ).order_by(
        KnowledgeBaseChunk.embedding.cosine_distance(query_embedding_pg)
    ).limit(limit).all()

    return [(chunk, score) for chunk, score in chunks]


async def get_hybrid_search_results(kb: KnowledgeBase, request: SearchRequest, db: Session) -> List[SearchScoreResult]:
    # 确定混合比例
    hybrid_ratio = kb.hybrid_ratio
    # 并行执行两种搜索
    text_results = get_text_search_results(db, kb.id, request.query, request.limit * 2)
    vector_results = await get_vector_search_results(db, kb.id, request.query, request.limit * 2)

    # 处理结果
    text_chunks, text_scores = zip(*text_results) if text_results else ([], [])
    vector_chunks, vector_scores = zip(*vector_results) if vector_results else ([], [])

    # 归一化分数
    text_scores_norm = normalize_scores(text_scores) if text_scores else []
    vector_scores_norm = normalize_scores(vector_scores) if vector_scores else []

    # 合并结果
    merged_results = merge_search_results(
        text_chunks=text_chunks,
        text_scores=text_scores_norm,
        vector_chunks=vector_chunks,
        vector_scores=vector_scores_norm,
        hybrid_ratio=hybrid_ratio
    )

    # 返回Top-K结果
    return [
        SearchScoreResult(
            content=chunk.content,
            file_name=chunk.file_name,
            file_id=chunk.file_id,
            chunk_index=chunk.chunk_index,
            score=score
        )
        for chunk, score in merged_results[:request.limit]
    ]


@router.post("/{knowledge_base_id}/search", response_model=List[SearchScoreResult])
async def hybrid_search(
        knowledge_base_id: uuid.UUID,
        request: SearchRequest,
        db: Session = Depends(get_db),
        Authorize: AuthJWT = Depends()
):
    """混合搜索API（pgroonga全文 + 向量）"""
    try:
        user_id = auth.decode_jwt_to_uid(Authorize)
        kb = query_knowledge_base(db, user_id, knowledge_base_id)
        return await get_hybrid_search_results(kb, request, db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during search")


def normalize_scores(scores: List[float]) -> List[float]:
    """归一化分数到[0,1]范围"""
    if not scores:
        return []

    scores_arr = np.array(scores)
    if np.all(scores_arr == scores_arr[0]):  # 所有分数相同
        return [0.5] * len(scores_arr)

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(scores_arr.reshape(-1, 1)).flatten()
    return normalized.tolist()


def merge_search_results(
        text_chunks: List[KnowledgeBaseChunk],
        text_scores: List[float],
        vector_chunks: List[KnowledgeBaseChunk],
        vector_scores: List[float],
        hybrid_ratio: float
) -> List[Tuple[KnowledgeBaseChunk, float]]:
    """合并两种搜索结果并计算加权分数"""
    chunk_map = {}

    # 添加文本结果
    for chunk, score in zip(text_chunks, text_scores):
        chunk_map[chunk.id] = {
            "chunk": chunk,
            "text_score": score,
            "vector_score": 0.0
        }

    # 添加向量结果
    for chunk, score in zip(vector_chunks, vector_scores):
        if chunk.id in chunk_map:
            chunk_map[chunk.id]["vector_score"] = score
        else:
            chunk_map[chunk.id] = {
                "chunk": chunk,
                "text_score": 0.0,
                "vector_score": score
            }

    # 计算混合分数
    results = []
    for data in chunk_map.values():
        hybrid_score = (hybrid_ratio * data["text_score"] +
                        (1 - hybrid_ratio) * data["vector_score"])
        results.append((data["chunk"], hybrid_score))

    # 按分数排序
    return sorted(results, key=lambda x: x[1], reverse=True)

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
        db: Session = Depends(get_db),
        Authorize: AuthJWT = Depends()
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
    user_id = auth.decode_jwt_to_uid(Authorize)
    kb = db.query(KnowledgeBase).filter(
        (KnowledgeBase.id == knowledge_base_id) &
         (KnowledgeBase.uploader_id == user_id)
    ).first()

    if not kb:
        raise HTTPException(
            status_code=404,
            detail=f"知识库{knowledge_base_id}不存在或没有权限修改"
        )
    try:

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
        kb.is_public = request.is_public

        db.commit()
        db.refresh(kb)

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


    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail=str(e))


@router.delete(
    "/{knowledge_base_id}",
    status_code=204,
    summary="删除知识库",
    description="删除知识库元数据及关联数据表记录（不处理实际文件删除）",
    responses={
        204: {"description": "成功删除"},
        403: {"description": "没有操作权限"},
        404: {"description": "知识库不存在"},
        500: {"description": "数据库操作失败"}
    }
)
async def delete_knowledge_base(
        knowledge_base_id: UUID,
        db: Session = Depends(get_db),
        Authorize: AuthJWT = Depends()
):
    """
    删除知识库API

    安全规则：
    - 只有知识库上传者可以删除
    - 自动级联删除关联分块数据
    - 数据库事务保证原子性

    注意：不会删除文件系统中的原始文件
    """
    user_id = auth.decode_jwt_to_uid(Authorize)

    try:
        # 查询知识库并验证权限（单次查询优化）
        kb = db.query(KnowledgeBase).filter(
            (KnowledgeBase.id == knowledge_base_id) &
            (KnowledgeBase.uploader_id == user_id)
        ).with_for_update().first()  # 加锁防止并发修改

        if not kb:
            raise HTTPException(
                status_code=404,
                detail="知识库不存在或没有删除权限"
            )

        # 批量删除关联分块（性能优化）
        chunk_count = db.query(KnowledgeBaseChunk) \
            .filter_by(knowledge_base_id=knowledge_base_id) \
            .delete(synchronize_session=False)

        # 删除主记录
        db.delete(kb)
        db.commit()

        logger.info(
            f"User {user_id} deleted KB {knowledge_base_id} "
            f"(removed {chunk_count} chunks)"
        )

        # 204 No Content 响应
        return {"description": "成功删除"}

    except HTTPException:
        raise  # 直接抛出已处理的HTTP异常
    except exc.SQLAlchemyError as e:
        logger.error(f"Database error deleting KB {knowledge_base_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="数据库操作失败"
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting KB {knowledge_base_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="服务器内部错误"
        )

@router.get(
    "/{knowledge_base_id}/files/{file_id}/download",
    summary="下载知识库中的文件",
    description="下载知识库中指定的文件，返回文件二进制数据",
    responses={
        200: {"description": "文件下载成功"},
        403: {"description": "没有权限访问该知识库或文件"},
        404: {"description": "知识库或文件不存在"},
        500: {"description": "服务器内部错误"}
    }
)
async def download_knowledge_base_file(
    knowledge_base_id: UUID,
    file_id: UUID,
    db: Session = Depends(get_db),
    Authorize: AuthJWT = Depends()
):
    """
    下载知识库中的文件API
    
    参数:
    - knowledge_base_id: 知识库唯一标识(UUID格式)
    - file_id: 文件唯一标识(UUID格式)
    
    返回:
    - 文件二进制数据，带有正确的Content-Type和Content-Disposition头
    
    错误码:
    - 403: 没有权限访问该知识库或文件
    - 404: 知识库或文件不存在
    - 500: 服务器内部错误
    """
    user_id = auth.decode_jwt_to_uid(Authorize)
    
    try:
        # 1. 验证用户对知识库的访问权限
        kb = query_knowledge_base(db, user_id, knowledge_base_id)
        
        # 2. 验证文件是否属于该知识库
        file_in_kb = db.query(KnowledgeBaseChunk).filter(
            KnowledgeBaseChunk.knowledge_base_id == knowledge_base_id,
            KnowledgeBaseChunk.file_id == file_id
        ).first()
        
        if not file_in_kb:
            raise HTTPException(
                status_code=404,
                detail=f"文件 {file_id} 不存在于知识库 {knowledge_base_id} 中"
            )
        
        # 3. 获取文件数据
        file_db = db.query(FileDB).filter(FileDB.id == file_id).first()
        
        if not file_db:
            raise HTTPException(
                status_code=404,
                detail=f"文件 {file_id} 不存在"
            )
        
        # 4. 返回文件响应
        # 正确处理文件名编码，使用RFC 5987规范进行URL编码

        encoded_filename = urllib.parse.quote(file_db.filename, encoding='utf-8')
        
        return Response(
            content=file_db.data,
            media_type=file_db.content_type,
            headers={
                # 提供两种格式的文件名，兼容不同浏览器
                "Content-Disposition": f'attachment; filename="{encoded_filename}"; filename*=UTF-8\'\'{encoded_filename}'
            }
        )
        
    except HTTPException:
        raise  # 直接抛出已知的HTTP异常
    except Exception as e:
        logger.error(f"下载文件失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

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

# @router.post("/{knowledge_base_id}/search", response_model=List[KnowledgeBaseSearchResult])
# def search_knowledge_base(
#         knowledge_base_id: UUID,
#         query: str,
#         db: Session = Depends(get_db)
# ):
#     """
#     搜索知识库内容
#
#     参数:
#     - knowledge_base_id: 要搜索的知识库ID
#     - query: 搜索关键词
#
#     功能:
#     - 使用PGroonga全文检索功能搜索知识库内容
#     - 仅搜索knowledge_base_chunks表中的content字段
#
#     返回:
#     - 匹配的文档片段列表，每个元素包含：
#         - content: 文档片段内容
#         - file_name: 来源文件名
#         - file_id: 来源文件ID
#         - chunk_index: 分块序号
#     """
#     return search_in_knowledge_base(db, knowledge_base_id, query)
#
#
# # 以下是参考文档中提到的其他API，可以按需添加：
#
# @router.put("/{knowledge_base_id}/rebuild", response_model=KnowledgeBaseResponse)
# def rebuild_knowledge_base(
#         knowledge_base_id: UUID,
#         rebuild_data: KnowledgeBaseRebuild,
#         background_tasks: BackgroundTasks,
#         db: Session = Depends(get_db)
# ):
#     """
#     重建知识库
#
#     参数:
#     - knowledge_base_id: 要重建的知识库ID
#     - rebuild_data: 包含新参数的请求体：
#         - chunk_size: 新的分块大小
#         - overlap_size: 新的重叠大小
#         - hybrid_ratio: 新的混合检索比例
#
#     功能:
#     1. 删除该知识库原有的所有分块数据
#     2. 更新知识库元数据中的参数
#     3. 状态重置为"building"
#     4. 异步重新处理文件分块
#
#     返回:
#     - 更新后的知识库基本信息
#     """
#     pass
#
#
# @router.delete("/{knowledge_base_id}")
# def delete_knowledge_base(
#         knowledge_base_id: UUID,
#         db: Session = Depends(get_db)
# ):
#     """
#     删除知识库
#
#     参数:
#     - knowledge_base_id: 要删除的知识库ID
#
#     功能:
#     1. 删除知识库元数据记录
#     2. 删除关联的所有分块数据
#     3. 不处理实际文件删除
#
#     返回:
#     - 204 No Content
#     """
#     pass
from datetime import datetime
import json
from typing import List
import uuid
from pydantic_ai import RunContext
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic_core import to_jsonable_python
from database import get_db_with
from database.model.knowledge_base import KnowledgeBaseChunk
from rest_model.knowledge_base import KnowledgeBaseBasicInfo, SearchRequest, SearchResult
from service.knowledge_base.knowledge_base_router import get_text_search_results, get_vector_search_results
from service.knowledge_base.knowledge_base_router import hybrid_search as hybrid_search_service

# --- 系统提示

def get_system_prompt() -> str:
    prompt = f'''该助手为DeepSeek Chat，由深度求索公司创造。
今天是{datetime.now().strftime("%Y年%m月%d日，星期%w").replace("星期0", "星期日")}。
对于创作类的问题（如写论文），请务必在正文的段落中引用对应的参考编号，不能只在文章末尾引用。
你需要解读并概括用户的题目要求，选择合适的格式，充分利用搜索结果并抽取重要信息，生成符合用户要求、极具思想深度、富有创造力与专业性的答案。
你的创作篇幅需要尽可能延长，对于每一个要点的论述要推测用户的意图，给出尽可能多角度的回答要点，且务必信息量大、论述详尽。
除非用户明确要求，否则你最终回答时使用的语言应该与用户提问一致。''' 
    return prompt

# --- 工具定义修改器

async def prepare_tool_def(ctx: RunContext[List[KnowledgeBaseBasicInfo]], tool_def: ToolDefinition) -> ToolDefinition | None:
    if len(ctx.deps) == 0:  # 知识库失效、或未提供知识库，则不提供检索工具
        return None
    if tool_def.name == "knowledge_base_metadata":
        tool_def.description += '你能够访问的知识库有：' + "\n".join([v.model_dump_json() for v in ctx.deps]) + '''
在必要时，使用合适的工具获取信息、提供答案，一次请求内可以使用多次相同的或不同的工具，当检索结果不理想时，考虑调整工具参数进行多次尝试，不要一次失败就放弃使用工具。
为了更加方便用户理解，你应该在调用工具之前告诉用户你的想法，例如“我应该……”，然后生成工具调用部分。
知识库中的所有内容并不都与用户的问题密切相关，你需要结合问题，对搜索结果进行甄别、筛选，并通过数次调整工具关键词关键词来获取更精确的结果。
在引用结果时，一定要使用[citation:file_id(uuid4):chunk_index(int)]的格式。
'''
    return tool_def

# --- 辅助函数

def validate_knowledge_base_id(knowledge_base_id: uuid.UUID, valid: List[KnowledgeBaseBasicInfo]):
    for kb in valid:
        if str(kb.knowledge_base_id) == str(knowledge_base_id):
            return True, ""
    return False, "错误的知识库ID"

# --- 工具定义

async def knowledge_base_metadata(void: None = None) -> None:
    '''
    knowledge_base_metadata工具不可被调用，仅用来向你告知可以在其它工具中使用的知识库ID。
    '''

async def keyword_search(ctx: RunContext[List[KnowledgeBaseBasicInfo]], knowledge_base_id: uuid.UUID, keyword: str, limit: int = 5):
    '''
    在知识库中搜索关键词，最多同时返回指定数量的结果（默认、至少5个，最多10个）。
    搜索引擎为关键词搜索引擎，将从知识库中进行准确匹配，使用空格分割多个关键词，但不要超过5个关键词。
    结果中，每个关键词都必须出现，所以如果关键词过多，将导致检索不到结果，确保查询keyword明确、具体、清晰。
    除非检索目标的关键词的字面含义明确而具体，比如特定人名，否则你应该使用语义搜索引擎进行检索。
    '''
    limit = max(5, min(limit, 10))
    valid, reason = validate_knowledge_base_id(knowledge_base_id, ctx.deps)
    if not valid:
        return reason
    with get_db_with() as db:
        result = get_text_search_results(db, knowledge_base_id, keyword, limit)
        if not result:
            return "没有找到相关的知识"
        else:
            return json.dumps([to_jsonable_python(SearchResult(
                    content=v.content,
                    file_id=v.file_id,
                    chunk_index=v.chunk_index,
                    file_name=v.file_name
                )) for v, score in result], ensure_ascii=False)
            
async def semantic_search(ctx: RunContext[List[KnowledgeBaseBasicInfo]], knowledge_base_id: uuid.UUID, keyword: str, limit: int = 5):
    '''
    在知识库中搜索相似的句子或段落，最多同时返回指定数量的结果（默认、至少5个，最多10个）。
    搜索引擎为语义搜索引擎，通过语义相似度检索，你应该提供连贯的段落的keyword作为查询输入。
    适合作为检索任务的入手点，获取广泛的、粗略的相关信息。
    '''
    limit = max(5, min(limit, 10))
    valid, reason = validate_knowledge_base_id(knowledge_base_id, ctx.deps)
    if not valid:
        return reason
    
    with get_db_with() as db:
        result = await get_vector_search_results(db, knowledge_base_id, keyword, limit)
        if not result:
            return "没有找到相关的知识"
        else:
            return json.dumps([to_jsonable_python(SearchResult(
                    content=v.content,
                    file_id=v.file_id,
                    chunk_index=v.chunk_index,
                    file_name=v.file_name
                )) for v, score in result], ensure_ascii=False)

async def hybrid_search(ctx: RunContext[List[KnowledgeBaseBasicInfo]], knowledge_base_id: uuid.UUID, keyword: str, limit: int = 5):
    '''
    在知识库中搜索关键词，最多同时返回指定数量的结果（默认、至少5个，最多10个）。
    搜索引擎为混合搜索引擎，结合文本相似度检索和关键词检索，你可以检索复杂、连续的文本，或简单的空格分割的关键词。
    '''
    limit = max(5, min(limit, 10))
    valid, reason = validate_knowledge_base_id(knowledge_base_id, ctx.deps)
    if not valid:
        return reason
    with get_db_with() as db:
        result = await hybrid_search_service(str(knowledge_base_id), SearchRequest(
            query=keyword,
            limit=limit
        ), db)
        if not result:
            return "没有找到相关的知识"
        else:
            return json.dumps([to_jsonable_python(SearchResult(**(v.model_dump()))) for v in result], ensure_ascii=False)

async def read_knowledge_base_content(ctx: RunContext[List[KnowledgeBaseBasicInfo]], file_id: uuid.UUID, chunk_index: List[int]):
    '''
    读取知识库的内容，返回指定file_id对应文件chunk_index处的内容，对于一个文件，它的chunk_index是从0开始的顺序增长的整数。
    你需要先调用其它检索工具来获取感兴趣的file_id和chunk_index。如果发现其它检索工具得到的结果，信息不完整，那么可以使用read_knowledge_base_content工具来查看相邻的chunk_index所存储的内容。
    建议一次性获取数个chunk_index的内容，以减小可能的错误。
    '''
    chunk_index = chunk_index[:10]
    with get_db_with() as db:
        result = db.query(KnowledgeBaseChunk).where(
            KnowledgeBaseChunk.file_id == file_id).where(KnowledgeBaseChunk.chunk_index.in_(chunk_index)).all()
        if not result:
            return "没有找到相关的知识"
        else:
            return json.dumps([to_jsonable_python(SearchResult(
                content=v.content,
                file_id=v.file_id,
                chunk_index=v.chunk_index,
                file_name=v.file_name
            )) for v in result], ensure_ascii=False)
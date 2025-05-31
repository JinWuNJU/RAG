from datetime import datetime
import json
from typing import List
import uuid
from pydantic_ai import RunContext, Agent
from pydantic_ai.tools import ToolDefinition
from pydantic_core import to_jsonable_python
from database import get_db_with
from database.model.knowledge_base import KnowledgeBaseChunk, KnowledgeBase
from rest_model.knowledge_base import KnowledgeBaseBasicInfo, SearchRequest, SearchResult
from service.knowledge_base.knowledge_base_router import get_text_search_results, get_vector_search_results, get_hybrid_search_results

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
    if tool_def.name == knowledge_base_metadata.__name__:
        tool_def.description += '你能够访问的知识库有：' + "\n".join([v.model_dump_json() for v in ctx.deps]) + '''
在必要时，使用合适的工具获取信息、提供答案，一次请求内可以使用多次相同的或不同的工具，当检索结果不理想时，考虑调整工具参数进行多次尝试，不要一次失败就放弃使用工具。
为了更加方便用户理解，你应该在调用工具之前告诉用户你的想法，例如"我应该……"，然后生成工具调用部分。
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

async def knowledge_base_metadata() -> None:
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
        kb = db.query(KnowledgeBase).get(knowledge_base_id)
        if not kb:
            return "知识库ID错误"
        result = await get_hybrid_search_results(kb, SearchRequest(query=keyword, limit=limit), db)
        if not result:
            return "没有找到相关的知识"
        else:
            return json.dumps([to_jsonable_python(SearchResult(**(v.model_dump()))) for v in result], ensure_ascii=False)

async def read_knowledge_base_content(ctx: RunContext[List[KnowledgeBaseBasicInfo]], file_id: uuid.UUID, chunk_index: List[int]):
    f'''
    读取知识库的内容，返回指定file_id对应文件chunk_index处的内容，对于一个文件，它的chunk_index是从1开始的顺序增长的整数。不要使用0作为chunk_index。
    file_id和chunk_index可以从其它工具的结果中获取。当用户希望查看某个文件的具体内容时，使用本工具。
    如果发现从其它检索工具得到的结果，可以推测出所需信息位于相邻的分块，那么可以使用本工具{read_knowledge_base_content.__name__}来查看相邻的chunk_index所存储的内容。
    不要用本工具来查看已经获得的chunk_index的内容，这样不会提供更多信息。
    建议一次性获取数个相邻chunk_index的内容，以获得更充足的信息。
    '''
    chunk_index = chunk_index[:10]
    if 0 in chunk_index:
        return "分块序号最小为1"
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
        
async def list_knowledge_base_files(ctx: RunContext[List[KnowledgeBaseBasicInfo]], knowledge_base_id: uuid.UUID, page: int = 0):
    f'''
    列出知识库中的文件列表。
    你可以用{list_knowledge_base_files.__name__}工具分页获取知识库下所有文件名，当用户进行宽泛的提问，如知识库里都有什么内容时；或者用户明确要求谈论某个文件时，可以使用本工具获取文件名列表。
    不要告诉用户file_id的具体值，这对他们来说没有意义。
    '''
    PAGE_SIZE = 40
    valid, reason = validate_knowledge_base_id(knowledge_base_id, ctx.deps)
    if not valid:
        return reason
    with get_db_with() as db:
        q = (
            db.query(
                KnowledgeBaseChunk.file_id,
                KnowledgeBaseChunk.file_name
            )
            .filter(KnowledgeBaseChunk.knowledge_base_id == knowledge_base_id)
            .group_by(
                KnowledgeBaseChunk.file_id,
                KnowledgeBaseChunk.file_name
            )
            .order_by(KnowledgeBaseChunk.file_id.desc())
        )
        files = q.offset(page * PAGE_SIZE).limit(PAGE_SIZE + 1).all()
        has_more = len(files) > PAGE_SIZE
        files = files[:PAGE_SIZE]
        result = [
            {
                'file_id': str(file_id),
                'file_name': file_name
            }
            for file_id, file_name in files
        ]
        return json.dumps({
            'files': result,
            'has_more': has_more
        }, ensure_ascii=False)
        
def config_agent(agent: Agent[List[KnowledgeBaseBasicInfo], str]):
    agent.system_prompt(
        get_system_prompt
    )
    agent.tool_plain(
        knowledge_base_metadata,
        prepare=prepare_tool_def
    )  # type: ignore
    agent.tool(
        keyword_search,
        prepare=prepare_tool_def
    )  # type: ignore
    agent.tool(
        semantic_search,
        prepare=prepare_tool_def
    )  # type: ignore
    # agent.tool(
    #     hybrid_search,
    #     prepare=prepare_tool_def
    # )  # type: ignore
    agent.tool(
        read_knowledge_base_content,
        prepare=prepare_tool_def
    )  # type: ignore
    agent.tool(
        list_knowledge_base_files,
        prepare=prepare_tool_def
    )  # type: ignore
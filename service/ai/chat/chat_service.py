import json
import os
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List

from fastapi import BackgroundTasks, HTTPException
from fastapi.logger import logger
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ToolCallPartDelta,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_core import to_jsonable_python
from sse_starlette import EventSourceResponse

from database import get_db_with
from database.model.chat import ChatHistoryDB, ChatMessageDB
from database.model.knowledge_base import KnowledgeBase, KnowledgeBaseChunk
from rest_model.chat.completions import MessagePayload
from rest_model.chat.history import ChatDetail, ChatHistory, ChatToolCallPart, ChatToolReturnPart
from rest_model.chat.sse import ChatBeginEvent, ChatEndEvent, ChatEvent, SseEventPackage, ToolCallEvent, ToolReturnEvent
from rest_model.knowledge_base import KnowledgeBaseBasicInfo,  SearchResult, SearchRequest
from service.ai.chat.service_base import BaseChatService
from dataclasses import dataclass, field
from typing import Callable, Optional

from service.knowledge_base.embedding import EmbeddingService
from service.knowledge_base.knowledge_base_router import hybrid_search


@dataclass
class LLM_Config:
    model_id: str
    api_key: Optional[str]
    endpoint: str
    system_prompt_fn: Callable[[], str] = field(repr=False)

    def system_prompt(self) -> str:
        return self.system_prompt_fn()

ChatLLM_Config = LLM_Config(
    model_id=os.environ.get("CHAT_LLM_MODEL_ID", "deepseek-v3-250324"),
    api_key=os.environ.get("CHAT_LLM_API_KEY"),
    endpoint=os.environ.get("CHAT_LLM_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/"),
    system_prompt_fn=lambda: f'''该助手为DeepSeek Chat，由深度求索公司创造。
今天是{datetime.now().strftime("%Y年%m月%d日，星期%w").replace("星期0", "星期日")}。
对于创作类的问题（如写论文），请务必在正文的段落中引用对应的参考编号，不能只在文章末尾引用。
你需要解读并概括用户的题目要求，选择合适的格式，充分利用搜索结果并抽取重要信息，生成符合用户要求、极具思想深度、富有创造力与专业性的答案。
你的创作篇幅需要尽可能延长，对于每一个要点的论述要推测用户的意图，给出尽可能多角度的回答要点，且务必信息量大、论述详尽。
在必要时，使用合适的工具获取信息、提供答案，一次请求内可以使用多次工具。为了更加方便用户理解，你应该在调用工具之前告诉用户你的想法，例如“我应该……”，然后生成工具调用部分。'''
)

TitleLLM_Config = LLM_Config(
    model_id=os.environ.get("TITLE_LLM_MODEL_ID", "glm-4-flash-250414"),
    api_key=os.environ.get("TITLE_LLM_API_KEY"),
    endpoint=os.environ.get("TITLE_LLM_API_ENDPOINT", "https://open.bigmodel.cn/api/paas/v4/"),
    system_prompt_fn=lambda: '''根据用户提问内容及助手的回答内容，生成对话标题。标题应简洁明了，能够准确概括对话的主题和内容。你的回答仅包含标题本身。标题不超过20个字。'''
)

@dataclass
class RagService:
    @staticmethod
    async def prepare_tool_def(ctx: RunContext[List[KnowledgeBaseBasicInfo]], tool_def: ToolDefinition) -> ToolDefinition | None:
        if len(ctx.deps) == 0:
            return None
        tool_def.description += '你能够访问的知识库有：' + "\n".join([v.model_dump_json() for v in ctx.deps])
        return tool_def
    
    @staticmethod
    async def search_keyword(ctx: RunContext[List[KnowledgeBaseBasicInfo]], knowledge_base_id: str, keyword: str, limit: int = 5):
        '''
        在知识库中搜索关键词，最多同时返回指定数量的结果（默认、至少5个，最多10个）。
        搜索引擎为混合搜索引擎，结合文本相似度检索和关键词检索，你可以检索复杂、连续的文本，或简单的空格分割的关键词。
        并非搜索结果的所有内容都与用户的问题密切相关，你需要结合问题，对搜索结果进行甄别、筛选，并通过数次调整关键词来获取更精确的结果。
        在引用结果时，一定要使用[citation:file_id:chunk_index]的格式。
        '''
        limit = max(5, min(limit, 10))
        for kb in ctx.deps:
            if str(kb.knowledge_base_id) == knowledge_base_id:
                break
        else:
            return "错误的知识库ID"
        with get_db_with() as db:
            result = await hybrid_search(knowledge_base_id, SearchRequest(
                query=keyword,
                limit=limit
            ), db, EmbeddingService())
            if not result:
                return "没有找到相关的知识"
            else:
                return json.dumps([to_jsonable_python(SearchResult(**(v.model_dump()))) for v in result], ensure_ascii=False)
            
    @staticmethod
    async def read_knowledge_base_content(ctx: RunContext[List[KnowledgeBaseBasicInfo]], file_id: uuid.UUID, chunk_index: List[int]):
        '''
        读取知识库的内容，返回指定file_id对应文件chunk_index处的内容，对于一个文件，它的chunk_index是从0开始的顺序增长的整数。
        你需要先调用search_keyword工具来获取感兴趣的file_id和chunk_index。
        建议一次性获取数个chunk_index的内容，以减小可能的错误。
        在引用结果时，一定要使用[citation:file_id:chunk_index]的格式。
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

class ChatService(BaseChatService):
    def __init__(self) -> None:
        super().__init__()
        model = OpenAIModel(ChatLLM_Config.model_id, 
                                provider=OpenAIProvider(
                                api_key=ChatLLM_Config.api_key, 
                                base_url=ChatLLM_Config.endpoint))
        self.agent = Agent(
            model,
            deps_type=List[KnowledgeBaseBasicInfo],
            result_type=str,
            system_prompt=ChatLLM_Config.system_prompt(),
        )
        self.agent.tool(
            RagService.search_keyword,
            prepare=RagService.prepare_tool_def
        )  # type: ignore
        self.agent.tool(
            RagService.read_knowledge_base_content,
            prepare=RagService.prepare_tool_def
        ) # type: ignore
        self.HISTORY_PAGE_SIZE = 20

        title_model = OpenAIModel(TitleLLM_Config.model_id,
                                        provider=OpenAIProvider(
                                        api_key=TitleLLM_Config.api_key, 
                                        base_url=TitleLLM_Config.endpoint))
        self.title_agent = Agent[None, str](
            title_model,
            result_type=str,
            system_prompt=TitleLLM_Config.system_prompt(),
        )
    
    async def get_chat(self, user_id: uuid.UUID, chat_id: str):
        with get_db_with() as db:
            chat_history: ChatHistoryDB | None = db.query(ChatHistoryDB).get(chat_id)
            if chat_history is None:
                raise HTTPException(status_code=404, detail="对话不存在")
            if chat_history.user_id != user_id:
                raise HTTPException(status_code=403, detail="没有权限访问该对话")
            chat_detail = ChatDetail.from_orm(chat_history)
            return chat_detail
    
    async def get_history(self, user_id: uuid.UUID, page: int = 1) -> List[ChatHistory]:
        with get_db_with() as db:
            chat_history = (
                db.query(ChatHistoryDB)
                .filter(ChatHistoryDB.user_id == user_id)
                .order_by(ChatHistoryDB.updated_at.desc())
                .offset((page - 1) * self.HISTORY_PAGE_SIZE)
                .limit(self.HISTORY_PAGE_SIZE)
                .all()
            )
        if not chat_history:
            return []
        return [ChatHistory.from_orm(history) for history in chat_history]

    
    async def _generate_root_message(self, payload: MessagePayload) -> ModelRequest:
        '''
        包含system prompt和用户消息的root message
        '''
        return ModelRequest(
            parts=[
                SystemPromptPart(
                    content=ChatLLM_Config.system_prompt()
                ),
                UserPromptPart(
                    content=payload.content
                )
            ]
        )
        
    async def _generate_user_message(self, payload: MessagePayload) -> ModelMessage:
        '''
        仅包含用户消息的message
        '''
        return ModelRequest(
            parts=[
                UserPromptPart(
                    content=payload.content
                )
            ]
        )
    
    async def _generate_chat_title(self, chat_id: uuid.UUID, user_query: str, assistant_answer: str):
        """
        生成聊天标题
        参数:
            user_query: 用户提问内容
            assistant_answer: 助手回答内容
        返回:
            生成的标题
        """
        user_prompt = f'''用户提问：```{user_query}```
助手回答：```{assistant_answer}```
        '''
        title = await self.title_agent.run(user_prompt=user_prompt)
        if (title := title.data.strip()) == "":
            return
        with get_db_with() as db:
            chat_history: ChatHistoryDB | None = db.query(ChatHistoryDB).get(chat_id)
            if chat_history is None:
                return
            chat_history.title = title
            db.commit()
        
    async def _generate_message_stream(self, 
                                       payload: MessagePayload, 
                                       history_id: uuid.UUID,
                                       user_message_id: uuid.UUID,
                                       chat_trace_list: deque[ModelMessage],
                                       generate_title: bool,
                                       background_tasks: BackgroundTasks,
                                       knowledge_base: List[KnowledgeBaseBasicInfo] = []
                                       ):
        """
        异步生成消息流，用于处理聊天消息的生成和工具调用。
        参数:
            payload: 用户消息负载
            history_id: 聊天记录ID
            user_message_id: 用户消息ID
            chat_trace_list: 聊天记录列表
            generate_title: 是否生成标题
            background_tasks: fastapi后台任务
            knowledge_base: 知识库列表
        """
        assistant_message_id=uuid.uuid4()
        yield SseEventPackage(
            ChatBeginEvent(
                chat_id=history_id,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id
            )
        )
        async with self.agent.iter(payload.content,  
                                   deps=knowledge_base,
                              message_history=list(chat_trace_list)) as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as request_stream:
                        async for event in request_stream:
                            if isinstance(event, PartStartEvent):
                                if event.part.part_kind == 'text' and event.part.content:
                                    yield SseEventPackage(
                                        ChatEvent(
                                            content=event.part.content
                                        )
                                    )
                            elif isinstance(event, PartDeltaEvent):
                                if isinstance(event.delta, TextPartDelta):
                                    if event.delta.content_delta:
                                        yield SseEventPackage(
                                            ChatEvent(
                                                content=event.delta.content_delta
                                            )
                                        )
                                elif isinstance(event.delta, ToolCallPartDelta):
                                    pass
                            elif isinstance(event, FinalResultEvent):
                                pass
                elif Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                yield SseEventPackage(
                                    ToolCallEvent(
                                        data=ChatToolCallPart(
                                            tool_name=event.part.tool_name,
                                            args=event.part.args
                                        )
                                    )
                                )
                            elif isinstance(event, FunctionToolResultEvent):
                                yield SseEventPackage(
                                    ToolReturnEvent(
                                        data=ChatToolReturnPart(
                                            tool_name=event.result.tool_name or '',
                                            content=to_jsonable_python(event.result.content)
                                        )
                                    )
                                )
                elif Agent.is_end_node(node):
                    pass
            if (result := run.result) is not None:
                new_messages = result.new_messages()
                # 生成标题
                if generate_title:
                    user_query = ""
                    assistant_answer = ""
                    for msg in new_messages:
                        for part in msg.parts:
                            if part.part_kind == "user-prompt":
                                user_query += str(part.content)
                            elif part.part_kind == "text":
                                assistant_answer += part.content
                    background_tasks.add_task(
                        self._generate_chat_title,
                        chat_id=history_id,
                        user_query=user_query,
                        assistant_answer=assistant_answer
                    )
                # 向数据库中存储ai助手的回答
                # 排除user prompt和system prompt，因为相关信息在请求ai之前已经在message_stream中添加到数据库了
                filtered_answer_output = filter(
                    lambda x: not any(p.part_kind == 'user-prompt' or p.part_kind == 'system-prompt' for p in x.parts),
                    new_messages
                )
                with get_db_with() as db:
                    db.add(ChatMessageDB(
                        id=assistant_message_id,
                        role="assistant",
                        part=list(filtered_answer_output),
                        chat_id=history_id,
                        parent_id=user_message_id,
                    ))
                    db.commit()
            else:
                logger.warning("Agent: query %s returned None", payload.content)
            yield SseEventPackage(
                ChatEndEvent()
            )

    async def message_stream(self, user_id: uuid.UUID, payload: MessagePayload, background_tasks: BackgroundTasks) -> EventSourceResponse:
        with get_db_with() as db:
            is_create_new_chat = False
            history_item = None
            parent_message = None
            chat_trace_list: deque[ModelMessage] = deque()

            if payload.chatId is not None:
                history_item: ChatHistoryDB | None = db.query(ChatHistoryDB).get(payload.chatId)
            if history_item is None:
                # 新建对话，当提供了chatId，但没有找到对应的history_item，或者没有提供chatId
                is_create_new_chat = True
                history_item = ChatHistoryDB(
                    id=uuid.uuid4(),
                    title=payload.content[:10],
                    user_id=user_id,
                    knowledge_base=payload.knowledgeBase if payload.knowledgeBase else None,
                )
                user_message = ChatMessageDB(
                    id=uuid.uuid4(),
                    role="user",
                    part=[await self._generate_root_message(payload)],
                    chat_history=history_item
                )
                db.add(history_item)
                db.commit()
            else:
                # 存在history_item
                chat_full_list: List[ChatMessageDB] = history_item.chat
                if payload.parentId is not None:
                    # 如果提供了parentId，尝试查找对应的消息
                    # 如果没有找到对应的消息，或者parentId不属于当前对话历史记录，抛出404错误
                    parent_message: ChatMessageDB | None = next(
                        (msg for msg in chat_full_list if msg.id == uuid.UUID(payload.parentId)), 
                        None
                    )
                    if parent_message is None:
                        raise HTTPException(
                            status_code=404,
                            detail="不正确的ParentId",
                        )
                if parent_message is None:
                    # 不提供parentId，即编辑了root message时
                    user_message = ChatMessageDB(
                        id=uuid.uuid4(),
                        role="user",
                        part=[await self._generate_root_message(payload)],
                        chat_history=history_item
                    )
                    db.add(user_message)
                    db.commit()
                else:
                    # 提供了parentId，表明是追问情况
                    if parent_message.role != "assistant":
                        raise HTTPException(
                            status_code=400,
                            detail="不正确的ParentId",
                        )
                    parent_message_id = parent_message.id
                    for chat in chat_full_list[::-1]:
                        if chat.id == parent_message_id:
                            chat_trace_list.extendleft(chat.part)
                            parent_message_id = chat.parent_id
                            if parent_message_id is None:
                                break
                    user_message = ChatMessageDB(
                        id=uuid.uuid4(),
                        role="user",
                        part=[await self._generate_user_message(payload)],
                        chat_history=history_item,
                        parent=parent_message
                    )
                    db.add(user_message)
                    history_item.updated_at = datetime.now(tz=timezone.utc)
                    db.commit()
            # 记录id来提供给聊天sse流
            history_id = history_item.id
            user_message_id = user_message.id
            knowledge_base_VOs = []
            if history_item.knowledge_base:
                knowledge_base_VOs = [KnowledgeBaseBasicInfo(
                        knowledge_base_id=v.id,
                        name=v.name,
                        description=v.description
                    ) for v in db.query(KnowledgeBase).filter(
                    KnowledgeBase.id.in_(history_item.knowledge_base)).all()]
            
        return EventSourceResponse(self._generate_message_stream(
            payload=payload, 
            history_id=history_id,
            user_message_id=user_message_id,
            chat_trace_list=chat_trace_list,
            generate_title=is_create_new_chat,
            background_tasks=background_tasks,
            knowledge_base=knowledge_base_VOs
        ))
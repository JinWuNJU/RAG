import os
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, List
from typing import Optional

from fastapi import BackgroundTasks, HTTPException
from fastapi.logger import logger
from pydantic_ai import Agent
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

import service.ai.tools.rag_agent_define as RagService
from database import get_db_with
from database.model.chat import ChatHistoryDB, ChatMessageDB
from database.model.knowledge_base import KnowledgeBase
from rest_model.chat.completions import MessagePayload
from rest_model.chat.history import ChatDetail, ChatHistory, ChatToolCallPart, ChatToolReturnPart
from rest_model.chat.sse import ChatBeginEvent, ChatEndEvent, ChatEvent, SseEventPackage, ToolCallEvent, ToolReturnEvent
from rest_model.knowledge_base import KnowledgeBaseBasicInfo
from service.ai.chat.service_base import BaseChatService
from utils import truncate_text_by_display_width


@dataclass
class LLM_Config:
    model_id: str
    api_key: Optional[str]
    endpoint: str


ChatLLM_Config = LLM_Config(
    model_id=os.environ.get("CHAT_LLM_MODEL_ID", "deepseek-v3-250324"),
    api_key=os.environ.get("CHAT_LLM_API_KEY"),
    endpoint=os.environ.get("CHAT_LLM_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/"),
)

TitleLLM_Config = LLM_Config(
    model_id=os.environ.get("TITLE_LLM_MODEL_ID", "glm-4-flash-250414"),
    api_key=os.environ.get("TITLE_LLM_API_KEY"),
    endpoint=os.environ.get("TITLE_LLM_API_ENDPOINT", "https://open.bigmodel.cn/api/paas/v4/")
)


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
        )
        self.agent.system_prompt(
            RagService.get_system_prompt
        )
        self.agent.tool_plain(
            RagService.knowledge_base_metadata,
            prepare=RagService.prepare_tool_def
        ) # type: ignore
        self.agent.tool(
            RagService.keyword_search,
            prepare=RagService.prepare_tool_def
        ) # type: ignore
        self.agent.tool(
            RagService.semantic_search,
            prepare=RagService.prepare_tool_def
        ) # type: ignore
        # self.agent.tool(
        #     RagService.hybrid_search,
        #     prepare=RagService.prepare_tool_def
        # )  # type: ignore
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
            system_prompt='根据用户提问内容及助手的回答内容，生成对话标题。标题应简洁明了，能够准确概括对话的主题和内容。你的回答仅包含标题本身。标题不超过20个字。',
        )
        
    @contextmanager
    def _get_chat_db(self, user_id: uuid.UUID, chat_id: str):
        '''
        获取ChatHistoryDB，并提供orm的事务上下文
        
        异常：   
        HTTP 404: 对话不存在   
        HTTP 403: 没有权限访问该对话
        '''
        with get_db_with() as db:
            try:
                chat_history: ChatHistoryDB | None = db.query(ChatHistoryDB).get(chat_id)
                if chat_history is None:
                    raise HTTPException(status_code=404, detail="对话不存在")
                if chat_history.user_id != user_id:
                    raise HTTPException(status_code=403, detail="没有权限访问该对话")
                yield chat_history
                db.commit()
            except Exception as e:
                db.rollback()
                raise e
    
    async def get_chat(self, user_id: uuid.UUID, chat_id: str):
        with self._get_chat_db(user_id, chat_id) as chat_history:
            chat_detail = ChatDetail.from_orm(chat_history)
            return chat_detail
    
    async def delete_chat(self, user_id: uuid.UUID, chat_id: str) -> bool:
        with self._get_chat_db(user_id, chat_id) as chat_history:
            chat_history.deleted = True
        return True
    
    async def get_history(self, user_id: uuid.UUID, page: int = 1) -> List[ChatHistory]:
        with get_db_with() as db:
            chat_history = (
                db.query(ChatHistoryDB)
                .filter(ChatHistoryDB.user_id == user_id)
                .filter(ChatHistoryDB.deleted == False)
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
                    content=RagService.get_system_prompt()
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
            msg_part_list: deque[ModelMessage] = deque()
            if payload.chatId is not None:
                history_item: ChatHistoryDB | None = db.query(ChatHistoryDB).get(payload.chatId)
            if history_item is None:
                # 新建对话，当提供了chatId，但没有找到对应的history_item，或者没有提供chatId
                is_create_new_chat = True
                history_item = ChatHistoryDB(
                    id=uuid.uuid4(),
                    title=truncate_text_by_display_width(payload.content, 20),
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
                if history_item.user_id != user_id:
                    raise HTTPException(status_code=403, detail="没有权限访问该对话")
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
                else: # 提供了parentId
                    # 提取到根消息的路径
                    chat_path: Deque[ChatMessageDB] = deque()
                    parent_message_id = parent_message.id
                    for chat in chat_full_list[::-1]:
                        if chat.id == parent_message_id:
                            chat_path.appendleft(chat)
                            parent_message_id = chat.parent_id
                            if parent_message_id is None:
                                break
                    if parent_message.role == 'assistant': # 追问或编辑提问
                        # 记录新的提问内容
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
                    elif parent_message.role == 'user': # 重新生成回答
                        # 提取最后一个user message的内容
                        # 并将它从trace_list中删除，使得ai看到的消息记录不包含重复的提问
                        user_message: ChatMessageDB = chat_path.pop()
                        payload.content = ''
                        for msg in user_message.part:
                            if msg.kind == 'request':
                                for part in msg.parts:
                                    if part.part_kind == 'user-prompt':
                                        payload.content += str(part.content)
                    else:
                        raise HTTPException(status_code=500) # 为了消除user_message Unbound警告
                    for chat in chat_path:
                        msg_part_list.extend(chat.part)
                        
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
            chat_trace_list=msg_part_list,
            generate_title=is_create_new_chat,
            background_tasks=background_tasks,
            knowledge_base=knowledge_base_VOs
        ))
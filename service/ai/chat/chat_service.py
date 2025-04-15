import os
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List

from fastapi import HTTPException
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

from database import get_db_with
from database.model.chat import ChatHistoryDB, ChatMessageDB
from rest_model.chat.completions import MessagePayload
from rest_model.chat.history import ChatHistory, ChatToolCallPart, ChatToolReturnPart
from rest_model.chat.sse import ChatBeginEvent, ChatEvent, SseEventPackage, ToolCallEvent, ToolReturnEvent
from service.ai.chat.service_base import BaseChatService
from dataclasses import dataclass, field
from typing import Callable, Optional


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
在必要时，使用合适的工具提供答案。为了更加方便用户理解，你应该在调用工具之前告诉用户你的想法，例如“我应该……”，然后生成工具调用部分。'''
)

TitleLLM_Config = LLM_Config(
    model_id=os.environ.get("TITLE_LLM_MODEL_ID", "glm-4-flash-250414"),
    api_key=os.environ.get("TITLE_LLM_API_KEY"),
    endpoint=os.environ.get("TITLE_LLM_API_ENDPOINT", "https://open.bigmodel.cn/api/paas/v4/"),
    system_prompt_fn=lambda: '''根据用户提问内容及助手的回答内容，生成对话标题。标题应简洁明了，能够准确概括对话的主题和内容。你的回答仅包含标题本身。标题不超过10个字。'''
)

@dataclass
class WeatherService:
    @staticmethod
    async def get_forecast(location: str, forecast_date: date) -> str:
        return f'The forecast in {location} on {forecast_date} is 24°C and sunny.'
    @staticmethod
    async def get_historic_weather(location: str, forecast_date: date) -> str:
        return (
            f'The weather in {location} on {forecast_date} was 18°C and partly cloudy.'
        )

class ChatService(BaseChatService):
    def __init__(self) -> None:
        super().__init__()
        model = OpenAIModel(ChatLLM_Config.model_id, 
                                provider=OpenAIProvider(
                                api_key=ChatLLM_Config.api_key, 
                                base_url=ChatLLM_Config.endpoint))
        self.agent = Agent[None, str](
            model,
            result_type=str,
            system_prompt=ChatLLM_Config.system_prompt(),
        )
        self.agent.tool_plain(
            WeatherService.get_forecast
        )
        self.HISTORY_PAGE_SIZE = 20

        # title_model = OpenAIModel(TitleLLM_Config.model_id,
        #                                 provider=OpenAIProvider(
        #                                 api_key=TitleLLM_Config.api_key, 
        #                                 base_url=TitleLLM_Config.endpoint))
        # self.title_agent = Agent[None, str](
        #     title_model,
        #     result_type=str,
        #     system_prompt=TitleLLM_Config.system_prompt(),
        # )
    
    async def get_chat(self, user_id: uuid.UUID, chat_id: str):
        raise NotImplementedError("get_chat method not implemented")
    
    async def get_history(self, user_id: uuid.UUID, page: int = 1):
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
        
    async def _generate_message_stream(self, 
                                       payload: MessagePayload, 
                                       history_id: uuid.UUID,
                                       user_message_id: uuid.UUID,
                                       chat_trace_list: deque[ModelMessage],
                                       history_item: ChatHistoryDB,
                                       parent_message: ChatMessageDB | None = None
                                       ):
        """
        异步生成消息流，用于处理聊天消息的生成和工具调用。
        参数:
            payload (MessagePayload): 包含用户提问内容的有效载荷。
            history_id (uuid.UUID): 对话所属的ChatHistoryDB对象的id。
            user_message_id (uuid.UUID): 新创建的用户消息的id。
            chat_trace_list (deque[ModelMessage]): 多轮对话的历史记录队列。
            history_item (ChatHistoryDB): 聊天历史数据库orm。
            parent_message (ChatMessageDB | None): 父消息对象，当创建新root message时不传递。·
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
                # 排除user prompt和system prompt，因为相关信息在请求ai之前已经在message_stream中添加到数据库了
                filtered_answer_output = filter(
                    lambda x: not any(p.part_kind == 'user-prompt' or p.part_kind == 'system-prompt' for p in x.parts),
                    result.new_messages()
                )
                with get_db_with() as db:
                    db.add(ChatMessageDB(
                        id=assistant_message_id,
                        role="assistant",
                        part=list(filtered_answer_output),
                        chat_history=history_item,
                        parent=parent_message
                    ))
                    db.commit()
                    db.refresh(history_item)
            else:
                logger.warning("Agent: query %s returned None", payload.content)

    async def message_stream(self, user_id: uuid.UUID, payload: MessagePayload):
        history_item = None
        parent_message = None
        chat_trace_list: deque[ModelMessage] = deque()
        with get_db_with() as db:
            if payload.chatId is not None:
                history_item: ChatHistoryDB | None = db.query(ChatHistoryDB).get(payload.chatId)
            if history_item is None:
                # 新建对话，当提供了chatId，但没有找到对应的history_item，或者没有提供chatId
                history_item = ChatHistoryDB(
                    id=uuid.uuid4(),
                    title=payload.content[:10],
                    user_id=user_id,
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
            
            return EventSourceResponse(self._generate_message_stream(
                payload=payload, 
                history_id=history_item.id,
                user_message_id=user_message.id,
                chat_trace_list=chat_trace_list,
                history_item=history_item,
                parent_message=user_message,
            ))
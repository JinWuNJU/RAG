import os
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List

from fastapi import HTTPException
from fastapi.logger import logger
from fastapi_jwt_auth2 import AuthJWT
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
from sqlalchemy.orm import Session
from sse_starlette import EventSourceResponse

from database.model.chat import ChatHistoryDB, ChatMessageDB
from rest_model.chat.completions import MessagePayload
from rest_model.chat.sse import ChatBeginEvent, ChatEvent, SseEventPackage, ToolCallEvent
from service.ai.chat.service_base import BaseChatService
from service.user import auth


class LLM_Config:
    model_id = os.environ.get("CHAT_LLM_MODEL_ID", "deepseek-v3-250324")
    api_key = os.environ.get("VOLCENGINE_API_KEY")
    endpoint = os.environ.get("VOLCENENGINE_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/")
    
    @staticmethod
    def system_prompt():
        current_date = datetime.now().strftime("%Y年%m月%d日，星期%w")
        return f'''该助手为DeepSeek Chat，由深度求索公司创造。
今天是{current_date.replace("星期0", "星期日")}。
在必要时，使用合适的工具提供答案。为了更加方便用户理解，你应该在调用工具之前告诉用户你的想法，例如“我应该……”，然后生成工具调用部分。'''
    
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
        self.model = OpenAIModel(LLM_Config.model_id, 
                                 provider=OpenAIProvider(
                                     api_key=LLM_Config.api_key, 
                                     base_url=LLM_Config.endpoint))
    
    async def get_chat(self, db: Session, Authorize: AuthJWT, chat_id: str):
        raise NotImplementedError("get_chat method not implemented")
    
    async def get_history(self, db: Session, Authorize: AuthJWT, page: int = 1):
        raise NotImplementedError("get_chat method not implemented")
    
    async def _generate_root_message(self, payload: MessagePayload) -> ModelRequest:
        '''
        包含system prompt和用户消息的root message
        '''
        return ModelRequest(
            parts=[
                SystemPromptPart(
                    content=LLM_Config.system_prompt()
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
                                       db: Session, 
                                       history_id: uuid.UUID,
                                       user_message_id: uuid.UUID,
                                       chat_trace_list: deque[ModelMessage],
                                       history_item: ChatHistoryDB,
                                       parent_message: ChatMessageDB | None = None,
                                       ):
        assistant_message_id=uuid.uuid4()
        yield SseEventPackage(
            ChatBeginEvent(
                chat_id=history_id,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id
            )
        )
        agent = Agent[None, str](
            self.model,
            result_type=str,
            system_prompt=LLM_Config.system_prompt(),
        )
        agent.tool_plain(
            WeatherService.get_forecast
        )
        async with agent.iter(payload.content,  
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
                                        name=event.part.tool_name,
                                        params=event.part.args,
                                        description=""
                                    )
                                )
                            elif isinstance(event, FunctionToolResultEvent):
                                pass
                                # f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}'
                elif Agent.is_end_node(node):
                    pass
            if (result := run.result) is not None:
                # 排除user prompt和system prompt，因为相关信息在请求ai之前已经在message_stream中添加到数据库了
                filtered_answer_output = filter(
                    lambda x: not any(p.part_kind == 'user-prompt' or p.part_kind == 'system-prompt' for p in x.parts),
                    result.new_messages()
                )
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

    async def message_stream(self, db: Session, Authorize: AuthJWT, payload: MessagePayload):
        user_id = auth.decode_jwt_to_uid(Authorize)
        history_item = None
        parent_message = None
        chat_trace_list: deque[ModelMessage] = deque()
        
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
            db=db,
            history_id=history_item.id,
            user_message_id=user_message.id,
            chat_trace_list=chat_trace_list,
            history_item=history_item,
            parent_message=user_message,
        ))
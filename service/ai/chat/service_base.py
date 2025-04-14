from abc import ABC, abstractmethod
from typing import List

from fastapi_jwt_auth2 import AuthJWT
from sqlalchemy.orm import Session
from sse_starlette import EventSourceResponse

from rest_model.chat.completions import MessagePayload
from rest_model.chat.history import ChatHistory


class BaseChatService(ABC):
    """聊天服务抽象基类，定义对外提供的接口"""
    
    @abstractmethod
    async def get_chat(self, db: Session, Authorize: AuthJWT, chat_id: str) -> ChatHistory | dict:
        """获取单个对话详情
        Args:
            chat_id: 对话ID
        Returns:
            包含对话详情的字典，不存在时返回空字典
        """
        pass
    
    @abstractmethod
    async def get_history(self, db: Session, Authorize: AuthJWT, page: int = 1) -> List[ChatHistory]:
        """获取对话历史（分页）
        Args:
            page: 页码，从1开始
        Returns:
            当前页的对话历史列表
        """
        pass
    
    @abstractmethod
    async def message_stream(self, db: Session, Authorize: AuthJWT, payload: MessagePayload) -> EventSourceResponse:
        """处理用户消息并返回SSE事件流
        Args:
            payload: 用户消息负载
        Returns:
            SSE事件流响应
        """
        pass
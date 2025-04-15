import uuid
from abc import ABC, abstractmethod
from typing import List

from sse_starlette import EventSourceResponse

from rest_model.chat.completions import MessagePayload
from rest_model.chat.history import ChatDetail, ChatHistory


class BaseChatService(ABC):
    """聊天服务抽象基类，定义对外提供的接口"""
    
    @abstractmethod
    async def get_chat(self, user_id: uuid.UUID, chat_id: str) -> ChatDetail | dict:
        """获取单个对话详情
        Args:
            chat_id: 对话ID
            user_id: 用户ID
        Returns:
            包含对话详情的字典，不存在时返回空字典
        """
        pass
    
    @abstractmethod
    async def get_history(self, user_id: uuid.UUID, page: int = 1) -> List[ChatHistory]:
        """获取对话历史（分页）
        Args:
            page: 页码，从1开始
            user_id: 用户ID
        Returns:
            当前页的对话历史列表
        """
        pass
    
    @abstractmethod
    async def message_stream(self, user_id: uuid.UUID, payload: MessagePayload) -> EventSourceResponse:
        """处理用户消息并返回SSE事件流
        Args:
            payload: 用户消息负载
            user_id: 用户ID
        Returns:
            SSE事件流响应
        """
        pass
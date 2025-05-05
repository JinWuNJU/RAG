import os

from fastapi import APIRouter, Depends, BackgroundTasks
from fastapi_jwt_auth2 import AuthJWT

from rest_model.chat.completions import MessagePayload
from service.ai.chat.chat_service import ChatService
from service.ai.chat.mock_chat_service import MockChatService
from service.ai.chat.mock_chat_service_legacy import MockChatServiceLegacy
from service.ai.chat.service_base import BaseChatService
from service.user import auth

router = APIRouter()

def get_chat_service() -> BaseChatService:
    if os.getenv("MOCKING_CHAT", "true").lower() == "false":
        return ChatService()
    return MockChatService()

chat_service = get_chat_service()

@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str, Authorize: AuthJWT = Depends()):
    """获取单个对话详情接口"""
    user_id = auth.decode_jwt_to_uid(Authorize)
    return await chat_service.get_chat(user_id, chat_id)

@router.get("/chats")
async def get_history(page: int = 1, Authorize: AuthJWT = Depends()):
    """获取对话历史接口"""
    user_id = auth.decode_jwt_to_uid(Authorize)
    return await chat_service.get_history(user_id, page)

@router.post("/completions")
async def message_stream(payload: MessagePayload, background_tasks: BackgroundTasks, Authorize: AuthJWT = Depends()):
    """处理用户消息并返回SSE事件流"""
    user_id = auth.decode_jwt_to_uid(Authorize)
    return await chat_service.message_stream(user_id, payload, background_tasks)

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, Authorize: AuthJWT = Depends()):
    """删除对话接口"""
    user_id = auth.decode_jwt_to_uid(Authorize)
    return await chat_service.delete_chat(user_id, chat_id)
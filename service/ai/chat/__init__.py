import os

from fastapi import APIRouter
from fastapi import Depends
from fastapi.logger import logger
from fastapi_jwt_auth2 import AuthJWT
from sqlalchemy.orm import Session

from database import get_db
from rest_model.chat.completions import MessagePayload
from service.ai.chat.chat_service import ChatService
from service.ai.chat.mock_chat_service import MockChatService
from service.ai.chat.service_base import BaseChatService

router = APIRouter()

def get_chat_service() -> BaseChatService:
    if os.getenv("MOCKING_CHAT", "true").lower() == "false":
        logger.info("Using real ChatService")
        return ChatService()
    return MockChatService()

chat_service = get_chat_service()

@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str, db: Session = Depends(get_db), Authorize: AuthJWT = Depends()):
    """获取单个对话详情接口"""
    return await chat_service.get_chat(db, Authorize, chat_id)

@router.get("/chats")
async def get_history(page: int = 1, db: Session = Depends(get_db), Authorize: AuthJWT = Depends()):
    """获取对话历史接口"""
    return await chat_service.get_history(db, Authorize, page)

@router.post("/completions")
async def message_stream(payload: MessagePayload, db: Session = Depends(get_db), Authorize: AuthJWT = Depends()):
    """处理用户消息并返回SSE事件流"""
    return await chat_service.message_stream(db, Authorize, payload)
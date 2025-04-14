from rest_model.chat.completions import MessagePayload
from service.ai.chat.service_base import BaseChatService


class ChatService(BaseChatService):
    """聊天服务实际实现，用于生产环境"""
    
    async def get_chat(self, chat_id: str):
        raise NotImplementedError("get_chat method not implemented")
    
    async def get_history(self, page: int = 1):
        raise NotImplementedError("get_chat method not implemented")
    
    async def message_stream(self, payload: MessagePayload):
        raise NotImplementedError("get_chat method not implemented")
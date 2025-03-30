import os
from openai import AsyncOpenAI
from typing import Optional

from pydantic import BaseModel, HttpUrl

class OpenAIConfig(BaseModel):
    api_key: str
    base_url: HttpUrl
    model: str
    timeout: Optional[float] = 30.0

class WrappedAsyncOpenAI:
    config: OpenAIConfig
    def __init__(self, config: OpenAIConfig):
        """初始化封装的AsyncOpenAI客户端
        
        Args:
            config: OpenAIConfig对象，包含API配置信息
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=str(config.base_url),
            timeout=config.timeout
        )
        self.default_model = config.model

    async def completionsCreate(self, **kwargs):
        '''
        使用默认模型创建一个新的对话
        '''
        return self.client.chat.completions.create(model=self.default_model, **kwargs)

    @property
    def chat(self):
        return self.client.chat

    
aiClient = WrappedAsyncOpenAI(
    OpenAIConfig(
        api_key=os.environ["VOLCENGINE_API_KEY"],
        base_url="https://ark.cn-beijing.volces.com/api/v3/",
        model="deepseek-v3-2450324"
    )
)
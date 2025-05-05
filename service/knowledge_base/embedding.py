import asyncio
import os
from typing import List, Optional
import numpy as np
from openai import AsyncOpenAI
from loguru import logger

from utils.window_ratelimiter import WindowRateLimiter


class EmbeddingService:
    instance = None
    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = EmbeddingService()
        return cls.instance
    def __init__(self):
        """
        初始化ARK嵌入服务，输出维度调整为1536
        """
        self.client = AsyncOpenAI(api_key=os.environ.get("EMB_API_KEY"), base_url=os.environ.get("EMB_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/"))
        self.model_name = os.environ.get("EMB_MODEL_ID", "doubao-embedding-text-240715")  # 原始模型输出2560维
        self.target_dim = 1536  # 目标维度
        # logger.info(f"初始化ARK嵌入服务，模型: {self.model_name}，目标维度: {self.target_dim}")

        # 进行限流，当前限制RPM=1200，与火山引擎的限制一致。
        # @see https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement?LLM=%7B%22PageSize%22%3A10%2C%22PageNumber%22%3A1%2C%22Filter%22%3A%7B%22VendorNameOrModelName%22%3A%5B%7B%22key%22%3A%22name%22%2C%22value%22%3A%22embedding%22%7D%5D%7D%7D&OpenTokenDrawer=false
        self.semaphore = asyncio.Semaphore(64)
        self.rate_limiter = WindowRateLimiter(max_requests=1200)

    def _normalize_and_slice(self, embedding: List[float]) -> np.ndarray:
        """归一化并截取到目标维度"""
        # 归一化处理
        norm = np.linalg.norm(embedding[:self.target_dim])
        normalized = [v / norm for v in embedding[:self.target_dim]]
        return np.array(normalized)

    async def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        生成1536维的嵌入向量
        :param text: 输入文本
        :return: 1536维的numpy数组
        """
        if not text.strip():
            return None

        try:
            async with self.semaphore:
                await self.rate_limiter.acquire()
                resp = await self.client.embeddings.create(
                    model=self.model_name,
                    input=[text]
                )
                return self._normalize_and_slice(resp.data[0].embedding)
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {str(e)}")
            return None

    async def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        批量生成1536维的嵌入向量（通过逐个调用embed_text实现）
        :param texts: 文本列表
        :return: 嵌入向量列表
        """
        if not texts:
            return []

        # 使用asyncio.gather并发调用embed_text
        embeddings = await asyncio.gather(
            *(self.embed_text(text) for text in texts)
        )
        return embeddings

    # async def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
    #     """
    #     批量生成1536维的嵌入向量
    #     :param texts: 文本列表
    #     :return: 嵌入向量列表
    #     """
    #     if not texts:
    #         return []
    #
    #     try:
    #         # 处理查询文本
    #
    #         resp = await self.client.embeddings.create(
    #             model=self.model_name,
    #             input=texts
    #         )
    #
    #         return [self._normalize_and_slice(item.embedding) for item in resp.data]
    #     except Exception as e:
    #         logger.error(f"批量生成嵌入向量失败: {str(e)}")
    #         return [None] * len(texts)
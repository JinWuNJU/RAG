import asyncio
import os
from typing import List, Optional
import numpy as np
from openai import AsyncOpenAI
from loguru import logger


class EmbeddingService:
    def __init__(self):
        """
        初始化ARK嵌入服务，输出维度调整为1536
        """
        self.client = AsyncOpenAI(api_key=os.environ.get("EMB_API_KEY"), base_url=os.environ.get("EMB_API_ENDPOINT"))
        self.model_name = os.environ.get("EMB_MODEL_ID", "")  # 原始模型输出2560维
        self.target_dim = 1536  # 目标维度
        # logger.info(f"初始化ARK嵌入服务，模型: {self.model_name}，目标维度: {self.target_dim}")

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
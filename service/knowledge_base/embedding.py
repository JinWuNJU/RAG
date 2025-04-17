import os
from typing import List, Optional
import numpy as np
from volcenginesdkarkruntime import Ark
from loguru import logger


class EmbeddingService:
    def __init__(self):
        """
        初始化ARK嵌入服务，输出维度调整为1536
        """
        self.client = Ark(api_key=os.environ.get("ARK_API_KEY"))
        self.model_name = os.environ.get("ARK_EMBEDDING_MODEL")  # 原始模型输出2048维
        self.target_dim = 1536  # 目标维度
        logger.info(f"初始化ARK嵌入服务，模型: {self.model_name}，目标维度: {self.target_dim}")

    def _normalize_and_slice(self, embedding: List[float]) -> np.ndarray:
        """归一化并截取到目标维度"""
        # 归一化处理
        norm = np.linalg.norm(embedding[:self.target_dim])
        normalized = [v / norm for v in embedding[:self.target_dim]]
        return np.array(normalized)

    def embed_text(self, text: str, is_query: bool = False) -> Optional[np.ndarray]:
        """
        生成1536维的嵌入向量
        :param text: 输入文本
        :param is_query: 是否为查询文本（需要添加instruction）
        :return: 1536维的numpy数组
        """
        if not text.strip():
            return None

        try:
            # 查询文本需要添加instruction
            input_text = f"为这个句子生成表示以用于检索相关文章：{text}" if is_query else text

            resp = self.client.embeddings.create(
                model=self.model_name,
                input=[input_text]
            )
            return self._normalize_and_slice(resp.data[0].embedding)
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {str(e)}")
            return None

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[Optional[np.ndarray]]:
        """
        批量生成1536维的嵌入向量
        :param texts: 文本列表
        :param is_query: 是否为查询文本
        :return: 嵌入向量列表
        """
        if not texts:
            return []

        try:
            # 处理查询文本
            inputs = [
                f"为这个句子生成表示以用于检索相关文章：{text}" if is_query else text
                for text in texts
            ]

            resp = self.client.embeddings.create(
                model=self.model_name,
                input=inputs
            )

            return [self._normalize_and_slice(item.embedding) for item in resp.data]
        except Exception as e:
            logger.error(f"批量生成嵌入向量失败: {str(e)}")
            return [None] * len(texts)
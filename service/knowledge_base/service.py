import asyncio
from typing import List

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session
import re

from database.model.file import FileDB
from database.model.knowledge_base import *
from uuid import UUID

from service.knowledge_base.embedding import EmbeddingService


class TextFileProcessor:
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = EmbeddingService()

    async def process_files(self, kb_id: UUID, file_ids: List[UUID]):
        """处理一批文本文件"""
        kb = self.db.query(KnowledgeBase).get(kb_id)
        if not kb:
            raise ValueError(f"知识库 {kb_id} 不存在")

        success_count = 0
        for file_id in file_ids:
            try:
                if await self._process_single_file(file_id, kb):
                    success_count += 1
            except Exception as e:
                logger.error(f"文件 {file_id} 处理失败: {str(e)}")
                continue

        # 更新知识库状态
        try:
            kb.status = "completed"
            if success_count == len(file_ids):  # 所有文件处理成功
                logger.success(f"所有文件处理完成，知识库 {kb_id} 状态已更新为 completed")
            else:
                logger.warning(f"部分文件处理完成，知识库 {kb_id} 状态为 partial_completed")

            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新知识库状态失败: {str(e)}")
            raise

        logger.success(f"处理完成: 成功 {success_count}/{len(file_ids)} 个文件")



    async def _process_single_file(self, file_id: UUID, kb: KnowledgeBase) -> bool:
        """处理单个文本文件"""
        # 1. 从数据库获取文本内容
        file_record = self.db.query(FileDB).filter_by(id=file_id).first()
        if not file_record:
            logger.warning(f"文件 {file_id} 不存在")
            return False

        # 2. 直接读取文本（假设data列是UTF-8编码的文本）
        try:
            text_content = file_record.data.decode('utf-8')
        except UnicodeDecodeError:
            logger.error(f"文件 {file_id} 不是有效的UTF-8文本")
            return False

        if not text_content.strip():
            logger.warning(f"文件 {file_id} 内容为空")
            return False

        # 3. 中文分块处理
        chunks = self._chunk_text(
            text_content,
            chunk_size=kb.chunk_size,
            overlap_size=kb.overlap_size
        )

        # 4. 生成嵌入向量
        embeddings = await self._generate_embeddings(chunks)

        # 5. 保存分块和嵌入向量
        self._save_chunks_with_embeddings(kb.id, file_id, file_record.filename, chunks, embeddings)
        return True

    async def _generate_embeddings(self, chunks: List[str]) -> List[Optional[np.ndarray]]:
        """异步生成嵌入向量"""
        try:
            return await self.embedding_service.embed_batch(chunks)
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")
            return [None] * len(chunks)

    def _chunk_text(self, text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """文本分块（改进版）
        - 主分块尽可能接近 chunk_size（按标点切分）
        - 重叠部分也按标点切分且长度 ≥ overlap_size
        - 避免产生过小的分块
        """
        # 预处理：标准化空白字符但保留段落分隔
        text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text)  # 单换行变空格
        text = re.sub(r'[ \t]+', ' ', text)  # 合并连续空白

        chunks = []
        pos = 0
        len_text = len(text)

        # 定义分割符优先级（可调整）
        SPLITTERS = [
            '\n\n',  # 段落分隔最高优先级
            '\n',  # 换行次之
            '。', '！', '？',  # 中文句子结束符
            '. ', '! ', '? ',  # 英文句子结束符（注意带空格）
            '；', '; ',  # 分号
            '，', ', ',  # 逗号
            ' ',  # 空格（最后的选择）
        ]

        while pos < len_text:
            # 1. 确定主分块结束点（尽可能接近 chunk_size）
            chunk_end = min(pos + chunk_size, len_text)

            # 查找最佳分割点（按优先级）
            split_pos = None
            for splitter in SPLITTERS:
                # 从后向前找最后一个分割符
                candidate = text.rfind(splitter, pos, chunk_end)
                if candidate > pos and (split_pos is None or candidate > split_pos):
                    split_pos = candidate + len(splitter)
                    # 如果找到足够大的分块（至少 chunk_size/2），可以提前终止
                    if split_pos - pos >= chunk_size * 0.5:
                        break

            # 如果找不到合适分割点或分块过小，尝试向前扩展
            if split_pos is None or (split_pos - pos) < chunk_size * 0.3:
                # 允许稍微超过 chunk_size 以找到合适的分割点
                chunk_end = min(pos + int(chunk_size * 1.5), len_text)
                for splitter in SPLITTERS:
                    candidate = text.rfind(splitter, pos, chunk_end)
                    if candidate > pos:
                        split_pos = candidate + len(splitter)
                        break

            # 最终确定主分块结束点
            chunk_end = split_pos if split_pos is not None else min(pos + chunk_size, len_text)

            # 2. 确定重叠开始点（确保 ≥ overlap_size 且按标点切分）
            overlap_start = max(pos, chunk_end - overlap_size * 2)  # 搜索范围扩大

            # 在重叠区域内查找分割点
            overlap_pos = None
            for splitter in SPLITTERS:
                candidate = text.find(splitter, overlap_start, chunk_end)
                if candidate != -1 and (chunk_end - candidate) >= overlap_size:
                    overlap_pos = candidate + len(splitter)
                    break

            # 最终确定重叠开始点
            next_pos = overlap_pos if overlap_pos is not None else max(pos, chunk_end - overlap_size)

            # 3. 添加分块（确保非空且足够大）
            chunk = text[pos:chunk_end].strip()
            if chunk and (not chunks or not self._is_redundant(chunk, chunks[-1], overlap_size)):
                chunks.append(chunk)

            # 4. 更新位置（确保前进）
            pos = max(next_pos, pos + chunk_size // 2)  # 至少前进50%块长度

        return chunks

    def _is_redundant(self,new_chunk: str, last_chunk: str, overlap: int) -> bool:
        """检查新块是否与上一块尾部过度重复"""
        if overlap <= 0:
            return False
        overlap_part = last_chunk[-overlap:]
        return new_chunk.startswith(overlap_part) and len(new_chunk) <= overlap * 1.5

    def _save_chunks_with_embeddings(self, kb_id: UUID, file_id: UUID, filename: str,
                                   chunks: List[str], embeddings: List[Optional[np.ndarray]]):
        """批量保存分块和嵌入向量到数据库"""
        if len(chunks) != len(embeddings):
            raise ValueError("分块数量和嵌入向量数量不匹配")

        db_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            db_chunk = KnowledgeBaseChunk(
                knowledge_base_id=kb_id,
                file_id=file_id,
                chunk_index=i + 1, # 从1开始
                content=chunk,
                file_name=filename,
                embedding=embedding.tolist() if embedding is not None else None
            )
            db_chunks.append(db_chunk)

        self.db.bulk_save_objects(db_chunks)
        self.db.commit()
        logger.info(f"文件 {filename} 已分块存储: {len(chunks)} 个分块 (其中 {sum(e is not None for e in embeddings)} 个有嵌入向量)")



# def search_in_knowledge_base(db: Session, kb_id: UUID, query: str) -> List[dict]:
#     """执行全文检索"""
#     return db.execute(
#         text("""
#         SELECT content, file_name, file_id, chunk_index
#         FROM knowledge_base_chunks
#         WHERE knowledge_base_id = :kb_id
#         AND content &@~ :query
#         LIMIT 20
#         """),
#         {"kb_id": str(kb_id), "query": query}
#     ).fetchall()
from typing import List

from loguru import logger
from sqlalchemy.orm import Session
import jieba
import re

from model.file.file import FileDB
from model.knowledge_base.knowledge_base_model import *

# 初始化jieba分词器
jieba.initialize()
jieba.setLogLevel(jieba.logging.INFO)


class TextFileProcessor:
    def __init__(self, db: Session):
        self.db = db

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
            if success_count == len(file_ids):  # 所有文件处理成功
                kb.status = "completed"
                logger.success(f"所有文件处理完成，知识库 {kb_id} 状态已更新为 completed")
            else:
                kb.status = "partial_completed"  # 可选：添加部分完成状态
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

        # 4. 保存分块
        self._save_chunks(kb.id, file_id, file_record.filename, chunks)
        return True

    def _chunk_text(self, text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """优化后的中文文本分块"""
        # 1. 清洗文本
        cleaned = re.sub(r'\s+', ' ', text).strip()

        # 2. 中文分词
        words = jieba.lcut(cleaned)

        # 3. 按词分块
        chunks = []
        start = 0
        total_words = len(words)

        while start < total_words:
            end = min(start + chunk_size, total_words)

            # 确保不截断句子（简单实现：遇到标点符号才分块）
            while (end < total_words and
                   not self._is_sentence_boundary(words[end - 1])):
                end += 1

            chunk = ''.join(words[start:end])
            chunks.append(chunk)
            start = max(end - overlap_size, start + 1)

        return chunks

    def _is_sentence_boundary(self, word: str) -> bool:
        """简单判断是否句子边界"""
        return word in {'。', '！', '？', '.', '!', '?'}

    def _save_chunks(self, kb_id: UUID, file_id: UUID, filename: str, chunks: List[str]):
        """批量保存分块到数据库"""
        db_chunks = [
            KnowledgeBaseChunk(
                knowledge_base_id=kb_id,
                file_id=file_id,
                chunk_index=i,
                content=chunk,
                file_name=filename
            )
            for i, chunk in enumerate(chunks)
        ]

        self.db.bulk_save_objects(db_chunks)
        self.db.commit()
        logger.info(f"文件 {filename} 已分块存储: {len(chunks)} 个分块")



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
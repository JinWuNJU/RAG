from typing import List

from loguru import logger
from sqlalchemy.orm import Session
import re

from model.file.file import FileDB
from model.knowledge_base.knowledge_base_model import *




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

        # 4. 保存分块
        self._save_chunks(kb.id, file_id, file_record.filename, chunks)
        return True

    def _chunk_text(self, text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """文本分块"""
        # 预处理：标准化空白字符但保留段落分隔
        text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text)  # 单换行变空格
        text = re.sub(r'[ \t]+', ' ', text)  # 合并连续空白

        chunks = []
        pos = 0
        len_text = len(text)

        while pos < len_text:
            # 计算本块理论终点
            end = min(pos + chunk_size, len_text)

            # 查找最佳分割点（三级回退策略）
            split_pos = None
            for candidate in [
                text.rfind('\n\n', pos, end),  # 1. 优先段落分隔
                text.rfind('\n', pos, end),  # 2. 次选行尾
                text.rfind('。', pos, end),  # 3. 句子结束
                text.rfind('！', pos, end),
                text.rfind('？', pos, end),
                text.rfind('.', pos, end),
                text.rfind('!', pos, end),
                text.rfind('?', pos, end),
                text.rfind('；', pos, end),  # 4. 分号
                text.rfind(';', pos, end),
                text.rfind('，', pos, end),  # 5. 逗号（最后选择）
                text.rfind(',', pos, end)
            ]:
                if candidate > pos and (split_pos is None or candidate > split_pos):
                    split_pos = candidate + 1  # 包含边界字符
                    break

            # 如果找不到合适边界且剩余文本过长，强制分割
            if split_pos is None and (end - pos) > chunk_size:
                split_pos = end

            # 确定最终分割点
            actual_split = split_pos if split_pos is not None else end

            # 获取当前块内容
            chunk = text[pos:actual_split].strip()
            if chunk:
                # 检查是否与前一块尾部重复（动态重叠控制）
                if not chunks or not self._is_redundant(chunk, chunks[-1], overlap_size):
                    chunks.append(chunk)

            # 更新位置（动态重叠调整）
            pos = max(
                actual_split - overlap_size,  # 理论重叠位置
                pos + chunk_size // 2  # 保证至少前进50%块长度
            ) if split_pos is not None else actual_split

        return chunks

    def _is_redundant(self,new_chunk: str, last_chunk: str, overlap: int) -> bool:
        """检查新块是否与上一块尾部过度重复"""
        if overlap <= 0:
            return False
        overlap_part = last_chunk[-overlap:]
        return new_chunk.startswith(overlap_part) and len(new_chunk) <= overlap * 1.5

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
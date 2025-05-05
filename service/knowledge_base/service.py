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
    
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = EmbeddingService.get_instance()

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

    def _find_split_point(self, text: str, start: int, end: int) -> Optional[int]:
        """
        在text的[start, end)范围内寻找合理的分割点
        规则：先从后向前寻找常见标点符号），如果找到则返回切分位置（切分点在标点后）
        如果没有找到，则返回None
        """
        # 定义标点（注意顺序不必与参考代码完全相同）
        punctuations = ['\n\n', '\n', '。', '！', '？', '. ', '! ', '? ', '；', '; ', '，', ', ', ' ']
        # 从标点列表中顺序寻找第一个匹配项（从后往前）
        for p in punctuations:
            idx = text.rfind(p, start, end)
            if idx != -1 and idx >= start:
                # 返回标点后的位置作为合理切分点
                return idx + len(p)
        return None
    
    def _chunk_text(self, text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """
        将文本根据预期分块长度(chunk_size)和重叠长度(overlap_size)进行切分。
        
        向后搜索切分点阶段：
        1. 每个分块至少为chunk_size的80%，在[pos + 0.8 * chunk_size, pos + chunk_size)范围内利用_find_split_point寻找合理切分点；
           若找不到，则采用完整的chunk_size作为分界直到文本末尾。
        2. 如果最后一块不足目标长度的80%，则将最后两块合并，在合并块中[0.45, 0.55]区间内寻找合理切分点，
           未找到则直接采用中点切分。
        
        向前搜索重叠点阶段：
        对除首块外的每一块，向前检测前一块的末尾，在20%误差范围内寻找理想的切分位置，以获取更自然的重叠文本，
        如果找不到，则直接定长切割。
        """
        # 预处理：标准化空白字符，但保留段落分隔
        text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text)  # 单换行合并成空格
        text = re.sub(r'[ \t]+', ' ', text)  # 合并连续空白

        chunks = []
        pos = 0
        len_text = len(text)
        threshold = 0.8 # 允许切分出最小 80% * chunk_size 的块

        # 向后搜索阶段：从文本前向后切分
        while pos < len_text:
            expected_end = min(pos + chunk_size, len_text)
            lower_bound = pos + int(chunk_size * threshold)
            if lower_bound >= len_text:
                # 如果剩余文本不足threshold * chunk_size，则将剩余部分作为一个块退出循环
                chunks.append(text[pos:].strip())
                break
            upper_bound = expected_end

            # 尝试在指定范围内寻找合理切分点
            split_point = self._find_split_point(text, lower_bound, upper_bound)
            if split_point is None:
                # 如果未找到，直接在预期长度处分割
                split_point = expected_end

            # 截取当前分块并清理空白
            chunk = text[pos:split_point].strip()
            if chunk:
                chunks.append(chunk)
            pos = split_point  # 更新当前起始位置

        # 最后一块检查：如果最后一个块不足目标长度threshold * chunk_size且存在前面块，则尝试合并最后两块重新切分
        if len(chunks) >= 2 and len(chunks[-1]) < (chunk_size * threshold):
            combined = chunks[-2] + " " + chunks[-1]
            lower_bound_comb = int(len(combined) * 0.45)
            upper_bound_comb = int(len(combined) * 0.55)
            new_split = self._find_split_point(combined, lower_bound_comb, upper_bound_comb)
            if new_split is None:
                # 未找到合理切分点则直接取中间位置
                new_split = len(combined) // 2
            # 更新最后两块为重新分割后的两部分
            chunks[-2] = combined[:new_split].strip()
            chunks[-1] = combined[new_split:].strip()

        # 向前搜索重叠点阶段：对除首块外的块进行合理的重叠调整
        adjusted_chunks = []
        adjusted_chunks.append(chunks[0])
        for i in range(1, len(chunks)):
            prev_chunk = adjusted_chunks[-1]
            # 理想重叠起始位置：前一块的末尾overlap_size字符
            ideal_overlap_start = max(0, len(prev_chunk) - overlap_size)
            # 允许20%误差范围
            error = int(overlap_size * 0.2)
            search_lower = max(0, ideal_overlap_start - error)
            search_upper = min(len(prev_chunk), ideal_overlap_start + error)
            # 在前一块的指定区域寻找合理切分点
            split_point = self._find_split_point(prev_chunk, search_lower, search_upper)
            if split_point is None:
                split_point = ideal_overlap_start  # 未找到则使用固定位置
            overlap_text = prev_chunk[split_point:]
            # 将重叠部分拼接到当前块前面
            new_chunk = overlap_text + chunks[i]
            adjusted_chunks.append(new_chunk)
        return adjusted_chunks

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
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
import os
import uuid
import asyncio
from unittest.mock import patch
from pydantic_ai import RunContext
import argparse
from tqdm import tqdm
import time
from sqlalchemy.exc import OperationalError
from contextlib import asynccontextmanager
import functools
from dotenv import load_dotenv

from database import get_db_with
from database.model.knowledge_base import KnowledgeBase
from rest_model.knowledge_base import KnowledgeBaseBasicInfo
from service.ai.tools.rag_agent_define import (
    keyword_search,
    semantic_search,
    hybrid_search,
    read_knowledge_base_content,
    config_agent,
    get_system_prompt,
    knowledge_base_metadata,
    prepare_tool_def
)

# 全局变量用于存储搜索结果
read_knowledge_base_content_results = []
keyword_search_results = []
semantic_search_results = []
hybrid_search_results = []

# 创建一个用于记录原始函数返回值到列表的包装器函数
def create_recorder(original_func: Callable, record_list: List) -> Callable:
    """
    创建一个包装器函数，将原始函数的返回值记录到列表中。
    """
    @functools.wraps(original_func)
    async def recorder_wrapper(*args, **kwargs):
        try:
            result = await original_func(*args, **kwargs)
            record_list.append(result)
            return result
        except Exception as e:
            tqdm.write(f"[DEBUG] {original_func.__name__} 出错: {str(e)}")  # 调试日志
            raise
    return recorder_wrapper

# 修改版 config_agent，只启用指定工具
def modified_config_agent(agent, tools: Optional[List[str]] = None) -> None:
    """修改版 config_agent，只启用指定的工具。"""
    if tools is None:
        tools = ['read_knowledge_base_content', 'semantic_search', 'keyword_search']
    
    # 使用全局的 recorder 函数
    agent.system_prompt(
        get_system_prompt
    )
    agent.tool_plain(
        knowledge_base_metadata,
        prepare=prepare_tool_def
    )  # type: ignore
    
    if 'read_knowledge_base_content' in tools:
        agent.tool(read_knowledge_base_content_recorder, prepare=prepare_tool_def)  # type: ignore
    if 'semantic_search' in tools:
        agent.tool(semantic_search_recorder, prepare=prepare_tool_def)  # type: ignore
    if 'keyword_search' in tools:
        agent.tool(keyword_search_recorder, prepare=prepare_tool_def)  # type: ignore
    if 'hybrid_search' in tools:
        agent.tool(hybrid_search_recorder, prepare=prepare_tool_def)  # type: ignore

# 创建记录器
read_knowledge_base_content_recorder = create_recorder(read_knowledge_base_content, read_knowledge_base_content_results)
keyword_search_recorder = create_recorder(keyword_search, keyword_search_results)
semantic_search_recorder = create_recorder(semantic_search, semantic_search_results)
hybrid_search_recorder = create_recorder(hybrid_search, hybrid_search_results)

# 保存原始函数的引用
original_read_knowledge_base_content = read_knowledge_base_content
original_keyword_search = keyword_search
original_semantic_search = semantic_search
original_hybrid_search = hybrid_search
original_config_agent = config_agent

# 创建 patches
patches = [
    patch('service.ai.tools.rag_agent_define.read_knowledge_base_content', new=read_knowledge_base_content_recorder),
    patch('service.ai.tools.rag_agent_define.keyword_search', new=keyword_search_recorder),
    patch('service.ai.tools.rag_agent_define.semantic_search', new=semantic_search_recorder),
    patch('service.ai.tools.rag_agent_define.hybrid_search', new=hybrid_search_recorder)
]

# 启动所有patches
for p in patches:
    p.start()

# 在 search 工具 patches 应用后，再 patch config_agent
# 创建一个闭包来保存命令行参数中的工具列表
def create_modified_config_agent(tools: List[str]):
    def _modified_config_agent(agent):
        return modified_config_agent(agent, tools)
    return _modified_config_agent

# 在 patches 应用后导入 ChatService
from service.ai.chat.chat_service import ChatService

# 加载环境变量
def load_env():
    """从 .env 文件加载环境变量。"""
    load_dotenv()

QUERY_CNT = 3  # 默认查询数量
# 从 JSON 文件读取查询
def read_queries(file_path: str) -> List[Dict]:
    """从 JSON 文件读取查询。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:QUERY_CNT]

# 处理单个查询，并将结果放入队列
async def process_single_query(
    query_data: Dict,
    chat_service: ChatService,
    knowledge_base: KnowledgeBase,
    queue: asyncio.Queue
) -> None:
    """处理单个查询，并将结果放入队列。"""
    response = None  # 初始化 response 变量
    try:
        query = query_data['query']
        ground_truth = query_data['answer']

        # 确保 knowledge_base_id 是有效的 UUID
        knowledge_base_vo = KnowledgeBaseBasicInfo(
            knowledge_base_id=knowledge_base.id,
            name=knowledge_base.name,
            description=knowledge_base.description
        )

        # 使用知识库作为依赖，从 chat service 获取响应
        max_tool_retries = 3
        tool_retry_delay = 2
        
        for retry in range(max_tool_retries):
            try:
                response = await chat_service.agent.run(
                        query, 
                        deps=[knowledge_base_vo]
                    )
                tqdm.write(f"[DEBUG] 获得响应: {response}")  # 调试日志
                break
            except Exception as e:
                if retry == max_tool_retries - 1:
                    raise
                tqdm.write(f"[DEBUG] 工具调用失败 (第 {retry + 1}/{max_tool_retries} 次): {str(e)}")
                await asyncio.sleep(tool_retry_delay)
        
        # 获取 response 的输出属性
        answer = response.data if response and hasattr(response, 'data') else str(response) if response else "未收到响应"
        
        # 从 recorder 列表获取记录的上下文
        retrieved_contexts = {
            'read_knowledge_base_content_results': read_knowledge_base_content_results[-1] if read_knowledge_base_content_results else None,
            'keyword_search_results': keyword_search_results[-1] if keyword_search_results else None,
            'semantic_search_results': semantic_search_results[-1] if semantic_search_results else None,
            'hybrid_search_results': hybrid_search_results[-1] if hybrid_search_results else None
        }
        
        result = {
            'query': query,
            'answer': answer,
            'ground_truth': ground_truth,
            'retrieved_contexts': retrieved_contexts
        }
        
        await queue.put(result)
    except asyncio.TimeoutError:
        tqdm.write(f"\n处理查询超时: {query_data.get('query', 'unknown')}")
        await queue.put(None)
    except Exception as e:
        tqdm.write(f"\n处理查询 '{query_data.get('query', 'unknown')}' 出错: {str(e)}")
        await queue.put(None)  # 用 None 表示出错

# 并发处理多个查询
async def process_queries(queries: List[Dict], chat_service: ChatService, knowledge_base: KnowledgeBase) -> List[Dict]:
    """并发处理多个查询。"""
    results = []
    queue = asyncio.Queue()
    max_concurrent_tasks = 200  # 并发数
    max_retries = 5  # 重试次数
    tool_retry_delay = 2  # 工具重试延迟时间（秒）
    
    # 创建信号量限制并发任务数
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    # 创建进度条，dynamic_ncols=True 保证进度条可见
    pbar = tqdm(total=len(queries), desc="处理查询", unit="query", dynamic_ncols=True)
    
    async def process_with_semaphore(query_data: Dict) -> None:
        async with semaphore:
            retries = 0
            while retries < max_retries:
                try:
                    # 设置工具调用的超时时间
                    await asyncio.wait_for(
                        process_single_query(query_data, chat_service, knowledge_base, queue),
                        timeout=300
                    )
                    pbar.update(1)
                    pbar.refresh()
                    break  # 如果成功，跳出重试循环
                except asyncio.TimeoutError:
                    retries += 1
                    if retries == max_retries:
                        tqdm.write(f"\n超时重试 {max_retries} 次后仍失败: {query_data.get('query', 'unknown')}")
                        await queue.put(None)
                        break
                    tqdm.write(f"\n超时，重试中 ({retries}/{max_retries})...")
                    await asyncio.sleep(tool_retry_delay)
                except OperationalError as e:
                    retries += 1
                    if retries == max_retries:
                        tqdm.write(f"\n数据库连接失败，重试 {max_retries} 次后仍失败: {query_data.get('query', 'unknown')}")
                        await queue.put(None)
                        break
                    tqdm.write(f"\n数据库连接失败，重试中 ({retries}/{max_retries})...")
                    await asyncio.sleep(tool_retry_delay)
                except Exception as e:
                    if "Tool exceeded max retries count" in str(e) or "ValidationError" in str(e):
                        retries += 1
                        if retries == max_retries:
                            tqdm.write(f"\n工具重试次数超限，已重试 {max_retries} 次: {query_data.get('query', 'unknown')}")
                            await queue.put(None)
                            break
                        tqdm.write(f"\n工具重试失败，重试中 ({retries}/{max_retries})...")
                        await asyncio.sleep(tool_retry_delay)
                    else:
                        tqdm.write(f"\n任务出错: 查询 '{query_data.get('query', 'unknown')}'，错误信息: {str(e)}")
                        await queue.put(None)
                        break
    
    # 为所有查询创建任务
    tasks = []
    for i, query_data in enumerate(queries):
        task = asyncio.create_task(process_with_semaphore(query_data))
        tasks.append(task)
    
    tqdm.write(f"\n已创建 {len(tasks)} 个任务用于处理查询")
    
    # 等待所有任务完成
    try:
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    await task  # 检查任务是否有异常
                except Exception as e:
                    tqdm.write(f"\n任务失败: {str(e)}")
    except Exception as e:
        tqdm.write(f"\n任务执行过程中出错: {str(e)}")
    finally:
        pbar.close()
    
    tqdm.write("\n所有任务已完成")
    
    # 从队列收集结果
    tqdm.write("正在从队列收集结果...")
    for _ in range(len(queries)):
        try:
            result = await queue.get()
            if result is not None:
                results.append(result)
        except Exception as e:
            tqdm.write(f"\n收集结果时出错: {str(e)}")
    
    tqdm.write(f"\n成功处理 {len(results)} / {len(queries)} 个查询")
    return results

# 数据库连接重试机制
default_max_db_retries = 3
default_db_retry_delay = 1
@asynccontextmanager
async def get_db_with_retry(max_retries: int = default_max_db_retries, retry_delay: int = default_db_retry_delay):
    """获取数据库连接，带重试机制。"""
    retries = 0
    while retries < max_retries:
        try:
            with get_db_with() as db:
                yield db
                return
        except OperationalError as e:
            retries += 1
            if retries == max_retries:
                raise
            tqdm.write(f"数据库连接失败，重试中 ({retries}/{max_retries})...")
            await asyncio.sleep(retry_delay)

# 主程序入口
async def main():
    global QUERY_CNT
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='RAG 评估工具')
    parser.add_argument('file_path', type=str, help='输入查询 JSON 文件路径')
    parser.add_argument('-o', '--output', type=str, help='输出 JSON 文件路径（可选）')
    parser.add_argument('--agent-tools', nargs='+', 
                       choices=['read_knowledge_base_content', 'semantic_search', 'keyword_search', 'hybrid_search'],
                       default=['read_knowledge_base_content', 'semantic_search', 'keyword_search'],
                       help='为 agent 启用的工具')
    parser.add_argument('--query-count', type=int, default=QUERY_CNT, help='要处理的查询数量 (默认: 200)')
    
    # 解析参数
    args = parser.parse_args()
    
    if not Path(args.file_path).exists():
        tqdm.write(f"错误: 文件 {args.file_path} 不存在")
        sys.exit(1)

    QUERY_CNT = args.query_count
    
    try:
        # 加载环境变量
        load_env()
        
        config_agent_patch = None
        try:
            # 使用命令行参数中的工具列表创建 modified_config_agent
            config_agent_patch = patch(
                'service.ai.tools.rag_agent_define.config_agent',
                new=create_modified_config_agent(args.agent_tools)
            )
            config_agent_patch.start()
            
            # 创建 chat service
            chat_service = ChatService()
            
            # 使用重试机制从数据库获取知识库
            async with get_db_with_retry() as db:
                # 在会话内查询知识库，并复制其数据以便后续使用
                kb_obj = db.query(KnowledgeBase).get(
                    uuid.UUID("ab1e4787-cdf2-4117-af12-89657177c00d")
                )
                if kb_obj:
                    # 通过复制数据将对象分离（或转为 dict）
                    knowledge_base = KnowledgeBase(
                        id=kb_obj.id,
                        name=kb_obj.name,
                        description=kb_obj.description
                    )
                else:
                    knowledge_base = None
                if not knowledge_base:
                    raise ValueError("未找到知识库")
                
            # 读取并处理查询
            queries = read_queries(args.file_path)
            tqdm.write(f"正在处理 {len(queries)} 个查询...")
            results = await process_queries(queries, chat_service, knowledge_base)
            
            # 输出结果
            output_json = json.dumps(results, indent=2, ensure_ascii=False)
            if args.output:
                # 如果指定输出路径则写入文件
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output_json)
                tqdm.write(f"结果已写入 {args.output}")
            else:
                # 未指定输出文件则打印到标准输出
                tqdm.write(output_json)
        finally:
            # 停止所有 patches
            for p in patches:
                p.stop()
            if config_agent_patch:
                config_agent_patch.stop()
    except Exception as e:
        tqdm.write(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

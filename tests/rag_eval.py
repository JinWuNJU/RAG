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

def create_recorder(original_func: Callable, record_list: List) -> Callable:
    """
    Create a wrapper function to record the original function's return value to a list.
    """
    @functools.wraps(original_func)
    async def recorder_wrapper(*args, **kwargs):
        try:
            result = await original_func(*args, **kwargs)
            record_list.append(result)
            return result
        except Exception as e:
            print(f"[DEBUG] Error in {original_func.__name__}: {str(e)}")  # Debug log
            raise
    return recorder_wrapper

def modified_config_agent(agent, tools: Optional[List[str]] = None) -> None:
    """Modified version of config_agent that only enables specified tools."""
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

def load_env():
    """Load environment variables from .env file."""
    load_dotenv()

def read_queries(file_path: str) -> List[Dict]:
    """Read queries from the JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[:10]

async def process_single_query(
    query_data: Dict,
    chat_service: ChatService,
    knowledge_base: KnowledgeBase,
    queue: asyncio.Queue
) -> None:
    """Process a single query and put the result in the queue."""
    response = None  # Initialize response variable
    try:
        query = query_data['query']
        ground_truth = query_data['answer']

        # 确保 knowledge_base_id 是有效的 UUID
        knowledge_base_vo = KnowledgeBaseBasicInfo(
            knowledge_base_id=knowledge_base.id,
            name=knowledge_base.name,
            description=knowledge_base.description
        )

        # Get response from the chat service with knowledge base as dependency
        max_tool_retries = 3
        tool_retry_delay = 2
        
        for retry in range(max_tool_retries):
            try:
                response = await asyncio.wait_for(
                    chat_service.agent.run(
                        query, 
                        deps=[knowledge_base_vo]
                    ),
                    timeout=30  # 30秒超时
                )
                print(f"[DEBUG] Got response: {response}")  # Debug log
                break
            except Exception as e:
                if retry == max_tool_retries - 1:
                    raise
                print(f"[DEBUG] Tool call failed (attempt {retry + 1}/{max_tool_retries}): {str(e)}")
                await asyncio.sleep(tool_retry_delay)
        
        # Access the output property of the response
        answer = response.output if response and hasattr(response, 'output') else str(response) if response else "No response received"
        
        # Get the recorded contexts from the recorder lists
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
        print(f"\nTimeout processing query: {query_data.get('query', 'unknown')}")
        await queue.put(None)
    except Exception as e:
        print(f"\nError processing query '{query_data.get('query', 'unknown')}': {str(e)}")
        await queue.put(None)  # Put None to indicate error

async def process_queries(queries: List[Dict], chat_service: ChatService, knowledge_base: KnowledgeBase) -> List[Dict]:
    """Process queries concurrently."""
    results = []
    queue = asyncio.Queue()
    max_concurrent_tasks = 10  # 降低并发数以减少压力
    max_retries = 5  # 增加重试次数
    tool_retry_delay = 2  # 工具重试延迟时间（秒）
    
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    # Create progress bar with dynamic_ncols=True to ensure it stays visible
    pbar = tqdm(total=len(queries), desc="Processing queries", unit="query", dynamic_ncols=True)
    
    async def process_with_semaphore(query_data: Dict) -> None:
        async with semaphore:
            retries = 0
            while retries < max_retries:
                try:
                    # 设置工具调用的超时时间
                    await asyncio.wait_for(
                        process_single_query(query_data, chat_service, knowledge_base, queue),
                        timeout=30
                    )
                    # 给数据库一些恢复时间
                    await asyncio.sleep(1)
                    pbar.update(1)
                    pbar.refresh()
                    break  # 如果成功，跳出重试循环
                except asyncio.TimeoutError:
                    retries += 1
                    if retries == max_retries:
                        print(f"\nTimeout after {max_retries} retries for query: {query_data.get('query', 'unknown')}")
                        await queue.put(None)
                        break
                    print(f"\nTimeout, retrying ({retries}/{max_retries})...")
                    await asyncio.sleep(tool_retry_delay)
                except OperationalError as e:
                    retries += 1
                    if retries == max_retries:
                        print(f"\nDatabase connection failed after {max_retries} retries for query: {query_data.get('query', 'unknown')}")
                        await queue.put(None)
                        break
                    print(f"\nDatabase connection failed, retrying ({retries}/{max_retries})...")
                    await asyncio.sleep(tool_retry_delay)
                except Exception as e:
                    if "Tool exceeded max retries count" in str(e) or "ValidationError" in str(e):
                        retries += 1
                        if retries == max_retries:
                            print(f"\nTool retry limit exceeded after {max_retries} attempts for query: {query_data.get('query', 'unknown')}")
                            await queue.put(None)
                            break
                        print(f"\nTool retry failed, retrying ({retries}/{max_retries})...")
                        await asyncio.sleep(tool_retry_delay)
                    else:
                        print(f"\nError in task for query '{query_data.get('query', 'unknown')}': {str(e)}")
                        await queue.put(None)
                        break
    
    # Create tasks for all queries
    tasks = []
    for i, query_data in enumerate(queries):
        task = asyncio.create_task(process_with_semaphore(query_data))
        tasks.append(task)
    
    print(f"\nCreated {len(tasks)} tasks for processing")
    
    # Wait for all tasks to complete
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
                    print(f"\nTask failed: {str(e)}")
    except Exception as e:
        print(f"\nError during task execution: {str(e)}")
    finally:
        pbar.close()
    
    print("\nAll tasks completed")
    
    # Collect results from queue
    print("Collecting results from queue...")
    for _ in range(len(queries)):
        try:
            result = await queue.get()
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"\nError collecting result: {str(e)}")
    
    print(f"\nSuccessfully processed {len(results)} out of {len(queries)} queries")
    return results

@asynccontextmanager
async def get_db_with_retry(max_retries: int = 3, retry_delay: int = 1):
    """Get database connection with retry mechanism."""
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
            print(f"Database connection failed, retrying ({retries}/{max_retries})...")
            await asyncio.sleep(retry_delay)

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='RAG Evaluation Tool')
    parser.add_argument('file_path', type=str, help='Path to the input JSON file containing queries')
    parser.add_argument('-o', '--output', type=str, help='Path to output JSON file (optional)')
    parser.add_argument('--agent-tools', nargs='+', 
                       choices=['read_knowledge_base_content', 'semantic_search', 'keyword_search', 'hybrid_search'],
                       default=['read_knowledge_base_content', 'semantic_search', 'keyword_search'],
                       help='Tools to enable for the agent')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not Path(args.file_path).exists():
        print(f"Error: File {args.file_path} does not exist")
        sys.exit(1)
    
    try:
        # Load environment variables
        load_env()
        
        config_agent_patch = None
        try:
            # 使用命令行参数中的工具列表创建 modified_config_agent
            config_agent_patch = patch(
                'service.ai.tools.rag_agent_define.config_agent',
                new=create_modified_config_agent(args.agent_tools)
            )
            config_agent_patch.start()
            
            # Create chat service
            chat_service = ChatService()
            
            # Get knowledge base from database with retry mechanism
            async with get_db_with_retry() as db:
                knowledge_base = db.query(KnowledgeBase).get(
                    uuid.UUID("57f2e448-82e9-49fb-aed8-8a7825819b18")
                )
                if not knowledge_base:
                    raise ValueError("Knowledge base not found")
                
                # Read and process queries
                queries = read_queries(args.file_path)
                print(f"Processing {len(queries)} queries...")
                results = await process_queries(queries, chat_service, knowledge_base)
                
                # Output results
                output_json = json.dumps(results, indent=2, ensure_ascii=False)
                if args.output:
                    # Write to file if output path is provided
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(output_json)
                    print(f"Results written to {args.output}")
                else:
                    # Print to stdout if no output file specified
                    print(output_json)
        finally:
            # Stop all patches
            for p in patches:
                p.stop()
            if config_agent_patch:
                config_agent_patch.stop()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

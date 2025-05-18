import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Callable
import os
import uuid
import asyncio
from unittest.mock import patch
from pydantic_ai import RunContext

from service.ai.chat.chat_service import ChatService
from database import get_db_with
from database.model.knowledge_base import KnowledgeBase
from rest_model.knowledge_base import KnowledgeBaseBasicInfo
from service.ai.tools.rag_agent_define import (
    keyword_search,
    semantic_search,
    hybrid_search
)

async def create_recorder(original_func: Callable, record_list: List, knowledge_base_vo: KnowledgeBaseBasicInfo, query: str) -> Callable:
    """
    Create a wrapper function to record the original function's return value to a list.
    
    Args:
        original_func: The original function to be wrapped.
        record_list: A list object where the original function's return value will be appended.
        knowledge_base_vo: The knowledge base info to use in the search.
        query: The query text to search for.
    
    Returns:
        A new function that receives the same parameters as original_func,
        executes original_func, adds the return value to record_list, and returns that value.
    """
    async def recorder_wrapper():
        ctx = RunContext[List[KnowledgeBaseBasicInfo]](
            deps=[knowledge_base_vo],
            model="gpt-3.5-turbo",  # 使用默认模型
            usage={},  # 空的使用统计
            prompt=""  # 空提示
        )
        
        # print(f"\nDebug: Calling {original_func.__name__}")
        # print(f"Debug: Knowledge base: {knowledge_base_vo.model_dump()}")
        # print(f"Debug: Query: {query}")
        # breakpoint()
        
        # Call original function with correct parameters
        result = await original_func(
            ctx,
            knowledge_base_vo.knowledge_base_id,
            query
        )
        
        # print(f"Debug: Result: {result}")
        record_list.append(result)

        return result
    
    await recorder_wrapper()
    return recorder_wrapper

def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"')

def read_queries(file_path: str) -> List[Dict]:
    """Read queries from the JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # For development, limit to first 3 queries
    return data[:3]

async def process_queries(queries: List[Dict], chat_service: ChatService, knowledge_base: KnowledgeBase) -> List[Dict]:
    """Process each query and get responses."""
    results = []
    
    try:
        for query_data in queries:
            query = query_data['query']
            ground_truth = query_data['answer']

            knowledge_base_vo = KnowledgeBaseBasicInfo(
                knowledge_base_id=knowledge_base.id,
                name=knowledge_base.name,
                description=knowledge_base.description
            )
            
            # Create recorders for each search function with current query context
            keyword_search_results = []
            semantic_search_results = []
            hybrid_search_results = []
            
            keyword_search_recorder = await create_recorder(keyword_search, keyword_search_results, knowledge_base_vo, query)
            semantic_search_recorder = await create_recorder(semantic_search, semantic_search_results, knowledge_base_vo, query)
            hybrid_search_recorder = await create_recorder(hybrid_search, hybrid_search_results, knowledge_base_vo, query)
            
            # Create patches for all search functions
            patches = [
                patch('service.ai.tools.rag_agent_define.keyword_search', side_effect=keyword_search_recorder),
                patch('service.ai.tools.rag_agent_define.semantic_search', side_effect=semantic_search_recorder),
                patch('service.ai.tools.rag_agent_define.hybrid_search', side_effect=hybrid_search_recorder)
            ]
            
            # Apply all patches
            for p in patches:
                p.start()
            
            try:
                # Get response from the chat service with knowledge base as dependency
                response = await chat_service.agent.run(query, deps=[knowledge_base_vo])
                # Access the output property of the response
                answer = response.output if hasattr(response, 'output') else str(response)
                
                # Collect all retrieved contexts
                retrieved_contexts = {
                    'keyword_search_results': keyword_search_results[-1] if keyword_search_results else None,
                    'semantic_search_results': semantic_search_results[-1] if semantic_search_results else None,
                    'hybrid_search_results': hybrid_search_results[-1] if hybrid_search_results else None
                }
                
                results.append({
                    'query': query,
                    'answer': answer,
                    'ground_truth': ground_truth,
                    'retrieved_contexts': retrieved_contexts
                })
            finally:
                # Stop all patches for this query
                for p in patches:
                    p.stop()
    
    except Exception as e:
        print(f"Error processing queries: {str(e)}")
        raise
    
    return results

async def main():
    if len(sys.argv) != 2:
        print("Usage: python -m tests.rag_eval <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    try:
        # Load environment variables
        load_env()
        
        # Create chat service
        chat_service = ChatService()
        
        # Get knowledge base from database
        with get_db_with() as db:
            knowledge_base = db.query(KnowledgeBase).get(
                uuid.UUID("7cf014b1-6ce0-444f-a1ae-842463b8f42b")
            )
            if not knowledge_base:
                raise ValueError("Knowledge base not found")
            
            # Read and process queries
            queries = read_queries(file_path)
            results = await process_queries(queries, chat_service, knowledge_base)
            
            # Output results
            print(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

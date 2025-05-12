#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模拟服务器用于测试
提供模拟的API响应，避免在测试中调用真实的外部服务
"""

import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# 创建FastAPI应用
app = FastAPI(title="RAG测试模拟服务器")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储模拟响应的字典
mock_responses: Dict[str, Any] = {}


@app.get("/")
async def root():
    """根路径处理程序"""
    return {"message": "RAG测试模拟服务器正在运行"}


@app.post("/v1/chat/completions")
async def mock_chat_completions(request: Request):
    """模拟聊天完成API"""
    body = await request.json()
    
    # 提取请求中的消息
    messages = body.get("messages", [])
    
    # 获取最后一条用户消息
    user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content")
            break
    
    # 根据用户消息生成响应
    response_content = generate_mock_response(user_message)
    
    # 构建响应
    response = {
        "id": "mock-chat-completion",
        "object": "chat.completion",
        "created": 1700000000,
        "model": body.get("model", "mock-model"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }
    
    # 添加延迟以模拟真实API调用
    await asyncio.sleep(0.2)
    
    return response


def generate_mock_response(user_message: Optional[str]) -> str:
    """根据用户消息生成模拟响应"""
    if not user_message:
        return "我没有收到任何消息"
    
    # 评估相关的响应 - 使用固定值
    if "评估" in user_message or "打分" in user_message:
        return "8.5"
    
    # 忠实度评估 - 使用与测试一致的固定值
    if "忠实" in user_message:
        return "8.7"  # 对应测试中的0.87
    
    # 相关性评估 - 使用与测试一致的固定值
    if "相关" in user_message:
        return "8.8"  # 对应测试中的0.88
    
    # 精确度评估 - 使用与测试一致的固定值
    if "精确" in user_message:
        return "9.1"  # 对应测试中的0.91
    
    # 默认响应
    return "这是一个模拟响应"


@app.post("/set_mock_response")
async def set_mock_response(request: Request):
    """设置模拟响应"""
    body = await request.json()
    endpoint = body.get("endpoint")
    response = body.get("response")
    
    if not endpoint or not response:
        return {"status": "error", "message": "缺少endpoint或response参数"}
    
    mock_responses[endpoint] = response
    return {"status": "success", "message": f"已为{endpoint}设置模拟响应"}


@app.post("/reset_mock_responses")
async def reset_mock_responses():
    """重置所有模拟响应"""
    mock_responses.clear()
    return {"status": "success", "message": "已重置所有模拟响应"}


@app.post("/{path:path}")
async def catch_all(path: str, request: Request):
    """捕获所有其他路径的请求并返回模拟响应"""
    # 尝试获取请求体
    try:
        body = await request.json()
    except:
        body = {}
    
    # 检查是否有预设的模拟响应
    if path in mock_responses:
        return mock_responses[path]
    
    # 默认响应
    return {
        "status": "success",
        "message": f"这是路径 {path} 的模拟响应",
        "received": body
    }


def start_mock_server(host: str = "127.0.0.1", port: int = 8000):
    """启动模拟服务器"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_mock_server() 
# RAG 后端

基于 FastAPI 和 PostgreSQL 的 RAG 后端服务。   
包含一个模拟服务器用于开发测试。

## 项目结构

- `server.py` - FastAPI Mock后端服务器
- `docker-compose.yaml` - PostgreSQL 容器配置，包含:
  - PostgreSQL 17
  - pg_search
  - pg_vector

## 环境要求

- 运行数据库容器
    - Docker 和 Docker Compose

- 运行 FastAPI 服务器
    - Python 3.10+
    - Python依赖包

## 使用步骤

- 数据库服务
    服务要求以下环境变量：
    - `POSTGRES_PASSWORD` - PostgreSQL 数据库密码

    
    **启动服务**
    ```bash
    make start_db
    ```

    数据库将在 `0.0.0.0:5432` 启动，用户名为 `postgres`，密码为 `${POSTGRES_PASSWORD}`。


- FastAPI 后端
    **启动服务**
    1. 安装 Python 依赖:
    ```bash
    pip install -r requirements.txt
    ```
    2. （若需要）在环境变量中填写LLM API KEY
    3. 启动服务器:
    ```bash
    make server
    ```

    服务器将在 `http://localhost:8000` 启动。

## 其他命令

- 停止数据库容器: `make stop_db`
- 清理数据库数据: `make clean_db`

## API 接口

- POST `/completions` - 流式对话接口
- GET `/chats` - 获取对话历史列表
- GET `/chats/{chat_id}` - 获取特定对话详情
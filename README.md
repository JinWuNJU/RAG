# RAG 后端

基于 FastAPI 和 PostgreSQL 的 RAG 后端服务。   
包含一个模拟服务器用于开发测试。

## 项目结构

- `server.py` - FastAPI 后端服务器
- `docker-compose.yaml` - PostgreSQL 容器配置，包含:
  - PostgreSQL 17
  - pgroonga
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

    **构建数据库容器**
    ```bash
    make build_image
    ```
    
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

    2. 在环境变量中填写必要的配置
        ```dot-env
        DB_HOST=数据库主机地址 (例如: 127.0.0.1)
        DB_PORT=数据库端口号 (例如: 5432)
        DB_NAME=数据库名称 (例如: RAG)
        DB_USER=数据库用户名 (例如: postgres)
        DB_PASS=数据库密码
        ```
        若需要，填写LLM API KEY

    3. 启动服务器:
        ```bash
        make server
        ```

    服务器将在 `http://localhost:8000` 启动。

## 其他命令

- 停止数据库容器: `make stop_db`
- 清理数据库数据: `make clean_db`

## 环境变量说明
```.env
POSTGRES_PASSWORD=容器数据库密码

DB_HOST=数据库主机地址 (例如: 127.0.0.1)
DB_PORT=数据库端口号 (例如: 5432)
DB_NAME=数据库名称 (例如: RAG)
DB_USER=数据库用户名 (例如: postgres)
DB_PASS=数据库密码

MOCKING_CHAT=true or false # 是否使用模拟llm聊天服务，默认true

# rag对话功能依赖
CHAT_LLM_MODEL_ID=聊天模型ID
VOLCENGINE_API_KEY=火山引擎API KEY
VOLCENENGINE_API_ENDPOINT=火山引擎API ENDPOINT

ZHIPU_API_KEY=智谱AI API KEY
```
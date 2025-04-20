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
POSTGRES_PASSWORD=容器内部署的数据库的密码

DB_HOST=数据库主机地址 (例如: 127.0.0.1)
DB_PORT=数据库端口号 (例如: 5432)
DB_NAME=数据库名称 (例如: RAG)
DB_USER=数据库用户名 (例如: postgres)
DB_PASS=数据库密码

MOCKING_CHAT=true or false # 是否使用模拟llm聊天服务，默认true
MOCKING_CHAT_LEGACY=true or false # 是否使用模拟服务的旧版本，默认false（新版本模拟了实际llm聊天的api，前端自4fe856d6起支持）


# rag对话功能依赖
CHAT_LLM_MODEL_ID=用于RAG对话的模型ID
CHAT_LLM_API_KEY=用于RAG对话的API KEY
CHAT_LLM_API_ENDPOINT=用于RAG对话的API ENDPOINT

TITLE_LLM_MODEL_ID=用于生成对话标题的模型ID
TITLE_LLM_API_KEY=用于生成对话标题的API KEY
TITLE_LLM_API_ENDPOINT=用于生成对话标题的API ENDPOINT

EMB_MODEL_ID=用于获取文本嵌入的模型ID
EMB_API_ENDPOINT=用于获取文本嵌入的API ENDPOINT
EMB_API_KEY=用于获取文本嵌入的API KEY

ZHIPU_API_KEY=智谱AI API KEY
```

##  部署

1. 部署到目标服务器采用shell 方式，以ssh方式登录到目标服务器，以scp 拷贝项目到目标地址/home/ubuntu/ragbackend，在目标路径中生成项目虚拟环境，且使用make server 启动服务，访问地址：http://172.29.4.37:8000。

2. 项目只要有新的修改提交就会触发构建

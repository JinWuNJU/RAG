# 使用 groonga/pgroonga 作为基础镜像
FROM groonga/pgroonga:4.0.1-debian-17

# 设置 pgvector 版本参数，方便后续修改
ARG PGVECTOR_VERSION=v0.8.0
# 设置 DEBIAN_FRONTEND 为非交互模式，避免 apt 安装过程中的提示中断构建
ARG DEBIAN_FRONTEND=noninteractive

# 安装构建 pg_vector 所需的依赖，编译安装 pg_vector，然后清理构建依赖和临时文件
# 将所有操作放在一个 RUN 指令中以减少镜像层数和大小
RUN apt-get update && \
    # 安装编译所需的包：git 用于克隆源码，build-essential 提供编译工具链 (make, gcc 等)
    # postgresql-server-dev-all 包含编译 PostgreSQL 扩展所需的头文件和 pg_config 工具
    apt-get install -y --no-install-recommends \
      git \
      build-essential \
      postgresql-server-dev-all && \
    \
    # 进入临时目录
    cd /tmp && \
    # 克隆指定版本的 pgvector 源码
    # 使用 --depth 1 进行浅克隆，减少下载量
    git clone --branch ${PGVECTOR_VERSION} --depth 1 https://github.com/pgvector/pgvector.git && \
    # 进入源码目录
    cd pgvector && \
    \
    # 编译 pg_vector
    # make 命令会自动调用 pg_config 查找 PostgreSQL 的安装路径和版本信息
    make && \
    # 安装 pg_vector
    make install && \
    \
    # 清理工作：删除源码目录
    cd / && \
    rm -rf /tmp/pgvector && \
    \
    # 清理工作：卸载仅在构建时需要的包，并清理 apt 缓存
    apt-get purge -y --auto-remove \
      git \
      build-essential \
      postgresql-server-dev-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建初始化脚本目录并添加 SQL 初始化脚本
RUN mkdir -p /docker-entrypoint-initdb.d
COPY init-db.sql /docker-entrypoint-initdb.d/

LABEL description="PostgreSQL with PGroonga and PGVector ${PGVECTOR_VERSION}"
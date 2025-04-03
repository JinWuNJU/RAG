-- 启用 pgvector 插件
CREATE EXTENSION IF NOT EXISTS vector;

-- 启用 pgroonga 插件
CREATE EXTENSION IF NOT EXISTS pgroonga;

-- 创建 users 表（带存在性检查）
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'users' AND schemaname = current_schema()) THEN
        CREATE TABLE users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL
        );

        -- 添加注释
        COMMENT ON TABLE users IS '系统用户表';
        COMMENT ON COLUMN users.username IS '用户名(唯一)';
        COMMENT ON COLUMN users.password IS '加密后的密码';

        RAISE NOTICE 'Table users created successfully';
    ELSE
        RAISE NOTICE 'Table users already exists, skipping creation';
    END IF;
END
$$;

-- 创建 knowledge_bases 表（带存在性检查）
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'knowledge_bases' AND schemaname = current_schema()) THEN
        CREATE TABLE knowledge_bases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(100) NOT NULL,
            description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            status VARCHAR(20) DEFAULT 'building' CHECK (status IN ('building', 'completed', 'failed')),
            chunk_size INTEGER NOT NULL CHECK (chunk_size > 0),
            overlap_size INTEGER NOT NULL CHECK (overlap_size >= 0),
            hybrid_ratio FLOAT DEFAULT 0.5 CHECK (hybrid_ratio BETWEEN 0 AND 1)
        );

        -- 添加注释
        COMMENT ON TABLE knowledge_bases IS '知识库元数据表';
        COMMENT ON COLUMN knowledge_bases.id IS '知识库唯一标识';
        COMMENT ON COLUMN knowledge_bases.name IS '知识库名称';
        COMMENT ON COLUMN knowledge_bases.status IS '状态: building/completed/failed';
        COMMENT ON COLUMN knowledge_bases.chunk_size IS '文档分块大小(字符数)';
        COMMENT ON COLUMN knowledge_bases.overlap_size IS '分块重叠大小(字符数)';
        COMMENT ON COLUMN knowledge_bases.hybrid_ratio IS '混合检索权重(0-1)';

        -- 创建索引
        CREATE INDEX idx_knowledge_bases_status ON knowledge_bases(status);

        RAISE NOTICE 'Table knowledge_bases created successfully';
    ELSE
        RAISE NOTICE 'Table knowledge_bases already exists, skipping creation';
    END IF;
END
$$;

-- 创建 knowledge_base_chunks 表（带存在性检查）
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'knowledge_base_chunks' AND schemaname = current_schema()) THEN
        CREATE TABLE knowledge_base_chunks (
            id SERIAL PRIMARY KEY,
            knowledge_base_id UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            file_id UUID NOT NULL,
            file_name VARCHAR(255) NOT NULL,
            chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
            content TEXT NOT NULL,
            embedding vector(1536)
        );

        -- 添加注释
        COMMENT ON TABLE knowledge_base_chunks IS '知识库分块内容表';
        COMMENT ON COLUMN knowledge_base_chunks.knowledge_base_id IS '关联的知识库ID';
        COMMENT ON COLUMN knowledge_base_chunks.file_id IS '原始文件ID';
        COMMENT ON COLUMN knowledge_base_chunks.embedding IS '文本向量嵌入(2048维)';

        -- 创建常规索引
        CREATE INDEX idx_chunks_knowledge_base ON knowledge_base_chunks(knowledge_base_id);
        CREATE INDEX idx_chunks_file ON knowledge_base_chunks(file_id);
        CREATE UNIQUE INDEX idx_chunks_uniq ON knowledge_base_chunks(knowledge_base_id, file_id, chunk_index);

        -- 创建PGroonga全文检索索引（支持中文）
        CREATE INDEX pgroonga_content_index ON knowledge_base_chunks USING pgroonga (content);

        -- 创建pgvector向量索引（IVFFlat算法）
        CREATE INDEX ivfflat_embedding_index ON knowledge_base_chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

        RAISE NOTICE 'Table knowledge_base_chunks created successfully';
    ELSE
        RAISE NOTICE 'Table knowledge_base_chunks already exists, skipping creation';
    END IF;
END
$$;
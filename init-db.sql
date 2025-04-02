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
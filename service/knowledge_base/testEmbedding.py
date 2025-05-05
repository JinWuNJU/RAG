import os
import numpy as np
from dotenv import load_dotenv
from embedding import EmbeddingService

# 加载环境变量
load_dotenv()


def test_embedding_service():
    print("=== 开始测试EmbeddingService ===")

    # 1. 测试初始化
    try:
        service = EmbeddingService.get_instance()
        print("✅ 初始化测试通过")
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        return

    # 2. 测试单个文本嵌入
    test_text = "自然语言处理是人工智能的重要分支"
    try:
        embedding = service.embed_text(test_text)
        if embedding is not None:
            print(f"✅ 单个文本嵌入测试通过，向量维度: {len(embedding)}")
            print(f"示例向量值: {embedding[:5]}...")
        else:
            print("❌ 单个文本嵌入返回了None")
    except Exception as e:
        print(f"❌ 单个文本嵌入失败: {str(e)}")

    # 3. 测试批量文本嵌入
    test_texts = [
        "深度学习需要大量计算资源",
        "Python是数据科学常用语言",
        "云计算提供了弹性计算能力"
    ]
    try:
        embeddings = service.embed_batch(test_texts)
        if len(embeddings) == len(test_texts):
            success_count = sum(1 for e in embeddings if e is not None)
            print(f"✅ 批量嵌入测试通过，成功数: {success_count}/{len(test_texts)}")
            for i, emb in enumerate(embeddings):
                if emb is not None:
                    print(f"文本{i + 1}向量维度: {len(emb)}")
        else:
            print(f"❌ 批量嵌入返回数量不匹配: {len(embeddings)} vs {len(test_texts)}")
    except Exception as e:
        print(f"❌ 批量嵌入失败: {str(e)}")

    # 4. 测试空文本处理
    try:
        empty_embedding = service.embed_text("")
        if empty_embedding is None:
            print("✅ 空文本处理测试通过")
        else:
            print("❌ 空文本处理失败")
    except Exception as e:
        print(f"❌ 空文本处理异常: {str(e)}")

    print("=== 测试完成 ===")


if __name__ == "__main__":
    test_embedding_service()
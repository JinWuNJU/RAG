from .user_router import router  # 显式暴露router对象

__all__ = ["router"]  # 定义允许通过 * 导入的内容
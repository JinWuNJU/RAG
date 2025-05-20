import functools
from typing import Callable, Any, Optional
from fastapi import Request, Depends
from fastapi_jwt_auth2 import AuthJWT
from slowapi import Limiter
from slowapi.util import get_remote_address
from .user.auth import decode_jwt_to_uid

# 创建limiter实例，使用IP地址作为默认key
limiter = Limiter(key_func=get_remote_address)

def rate_limit(
    ip_limit: str = "60/minute", 
    user_limit: str = "30/minute"
) -> Callable:
    """
    组合限流装饰器，先进行IP限流，再进行用户ID限流
    
    参数:
        ip_limit: IP限流强度，默认为每分钟60次
        user_limit: 用户ID限流强度，默认为每分钟30次
    
    返回:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        # 应用IP限流
        @limiter.limit(ip_limit)
        @functools.wraps(func)
        async def ip_limited_wrapper(
            request: Request,
            *args: Any,
            **kwargs: Any
        ) -> Any:
            # 提取AuthJWT对象
            auth_jwt = kwargs.get('Authorize')
            
            # 如果存在AuthJWT对象，再应用用户ID限流
            if auth_jwt and isinstance(auth_jwt, AuthJWT):
                try:
                    # 获取用户ID作为限流key
                    user_id = decode_jwt_to_uid(auth_jwt)
                    user_key = f"user:{user_id}"
                    
                    # 检查用户限流
                    limiter._check_request_limit(
                        request=request,
                        key_func=lambda: user_key,
                        limit_value=user_limit
                    )
                except Exception:
                    # 如果无法获取用户ID或者限流检查失败，继续处理
                    pass
            
            # 执行原始函数
            return await func(request, *args, **kwargs)
        
        return ip_limited_wrapper
    
    return decorator 
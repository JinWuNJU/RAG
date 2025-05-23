import functools
from typing import Callable, Any, Optional
from fastapi import HTTPException, Request, Depends
from fastapi.routing import APIRoute
from fastapi_jwt_auth2 import AuthJWT
from limits import strategies, storage, parse
from .user.auth import decode_jwt_to_uid

strategy = strategies.SlidingWindowCounterRateLimiter(storage=storage.MemoryStorage())

def rate_limit(
    ip_limit: str = "60/minute", 
    user_limit: str | None = None
) -> Callable:
    """
    组合限流装饰器，先进行IP限流，再进行用户ID限流
    要求endpoint函数参数中必须包含fastapi.Request实例
    需要应用于FastAPI的路由函数之后

    参数:
        ip_limit: IP限流强度，默认为每分钟60次
        user_limit: 用户ID限流强度，默认不开启

    返回:
        装饰后的函数
    """
    ip_limit_item = parse(ip_limit)
    user_limit_item = None
    if user_limit is not None:
        user_limit_item = parse(user_limit)

    def decorator(func: Callable[..., Any]) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            request: Optional[Request] = None
            auth: Optional[AuthJWT] = None
            for arg in kwargs.values():
                if isinstance(arg, Request):
                    request = arg
                if isinstance(arg, AuthJWT):
                    auth = arg
            # IP限流检查
            if request is None:
                raise TypeError("Endpoint argument requires a fastapi.Request instance")
            
            route: APIRoute | None = request.scope.get("route")
            if route is None:
                namespace = func.__module__ + "." + func.__qualname__
            else:
                namespace = route.path

            if request.client is not None:
                ip = request.client.host
            else:
                ip = "unknown"
            
            if not strategy.hit(ip_limit_item, namespace, ip):
                raise HTTPException(
                    status_code=429
                )
            # 用户ID限流检查
            if auth is not None and user_limit_item is not None:
                user_id = decode_jwt_to_uid(auth)
                if user_id is None:
                    raise HTTPException(
                        status_code=401,
                        detail="用户未登录"
                    )
                
                if not strategy.hit(user_limit_item, namespace, str(user_id)):
                    raise HTTPException(
                        status_code=429
                    )
            return await func(*args, **kwargs)
        return wrapper

    return decorator 
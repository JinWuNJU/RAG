from pydantic import BaseModel

class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(UserLogin):
    pass  # 注册可能扩展额外字段

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
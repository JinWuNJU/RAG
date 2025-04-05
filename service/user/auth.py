from datetime import timedelta
from uuid import UUID
from fastapi import HTTPException, status
from passlib.context import CryptContext
from fastapi_jwt_auth2 import AuthJWT

from database.model.user import User

# 密码哈希配置
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT配置
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str):
    return pwd_context.hash(password)

def gen_user_jwt(Authorize: AuthJWT, db_user: User):
    return Authorize.create_access_token(
        subject=str(db_user.id),
        expires_time=timedelta(days=7)
    )
    
def decode_jwt_to_uid(Authorize: AuthJWT):
    try:
        Authorize.jwt_required()
        user_id: str = Authorize.get_jwt_subject()
        return UUID(user_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登陆状态失效",
        )
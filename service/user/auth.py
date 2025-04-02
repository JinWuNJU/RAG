from passlib.context import CryptContext


# 密码哈希配置
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT配置
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str):
    return pwd_context.hash(password)


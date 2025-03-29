import os
from datetime import timedelta, datetime

from fastapi_jwt_auth2 import AuthJWT
from fastapi_jwt_auth2.exceptions import AuthJWTException
from pydantic import BaseModel

from service import chat
from fastapi.middleware.cors import CORSMiddleware

from service.file import file_router
from fastapi import FastAPI, Depends, HTTPException, status,Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import models
import schemas
import auth
from database import get_db
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(file_router.router)

class Settings(BaseModel):
    authjwt_secret_key: str = os.getenv("JWT_SECRET_KEY")




@AuthJWT.load_config
def get_config():
    return Settings()


@app.exception_handler(AuthJWTException)
def authjwt_exception_handler(request: Request, exc: AuthJWTException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message}
    )
# 注册接口
@app.post("/user/register", response_model=schemas.TokenResponse)
async def register(user: schemas.UserCreate,Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    # 检查用户名是否已存在
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已注册"
        )

    # 创建用户
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # 直接登录返回令牌
    # access_token = Authorize.create_access_token(subject=user.username)
    access_token = Authorize.create_access_token(
        subject=user.username,
        expires_time=timedelta(days=7)
    )
    return {"access_token": access_token}



# 登录接口
# 用户登录接口
@app.post("/user/login", response_model=schemas.TokenResponse)
async def login(user: schemas.UserLogin,Authorize: AuthJWT = Depends(), db: Session = Depends(get_db)):
    # 查询用户
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or not auth.verify_password(user.password, db_user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="不正确的用户名或密码",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 生成 JWT
    # access_token = Authorize.create_access_token(subject=user.username)
    access_token = Authorize.create_access_token(
        subject=user.username,
        expires_time=timedelta(days=7)
    )
    print(access_token)
    return {"access_token": access_token}



@app.get('/user/test')
def test_jwt1(Authorize: AuthJWT = Depends()):
    # 验证JWT令牌
    Authorize.jwt_required()
    current_user = Authorize.get_jwt_subject()
    return {
        "status": "JWT验证成功",
        "current_user": current_user,
        "test_passed": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
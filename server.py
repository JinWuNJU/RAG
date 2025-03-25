from service import chat
from fastapi.middleware.cors import CORSMiddleware

from service.file import file_router
from fastapi import FastAPI, Depends, HTTPException, status
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


# 注册接口
@app.post("/user/register", response_model=schemas.TokenResponse)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
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
    access_token = auth.create_access_token(
        data={"sub": str(db_user.id)}
    )
    return {"access_token": access_token, "token_type": "bearer"}


# 登录接口
# 用户登录接口
@app.post("/user/login", response_model=schemas.TokenResponse)
async def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    # 查询用户
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or not auth.verify_password(user.password, db_user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="不正确的用户名或密码",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 生成 JWT
    access_token = auth.create_access_token(
        data={"sub": str(db_user.id)}
    )
    return {"access_token": access_token, "token_type": "bearer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_jwt_auth2.exceptions import AuthJWTException

from service.ai import chat
from service.evaluation import router as evaluation_router
from service.file import file_router
from service.knowledge_base import knowledge_base_router
from service.user import user_router
from service.user.user_router import authjwt_exception_handler

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
app.include_router(user_router.router)

app.include_router(knowledge_base_router.router)

app.include_router(evaluation_router.router)

app.add_exception_handler(AuthJWTException, authjwt_exception_handler)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

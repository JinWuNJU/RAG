from pydantic import BaseModel
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from service import chat
from fastapi.middleware.cors import CORSMiddleware

from service.file import file_router

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
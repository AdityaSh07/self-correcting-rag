from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from . import models
from .database import engine, SessionLocal
from .routers import user, auth, chatbot
from .config import settings

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/", include_in_schema=False)
async def serve_login_page():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/signup", include_in_schema=False)
async def serve_signup_page():
    return FileResponse(FRONTEND_DIR / "signup.html")


@app.get("/chat", include_in_schema=False)
async def serve_chat_page():
    return FileResponse(FRONTEND_DIR / "chat.html")


app.include_router(auth.router)
app.include_router(user.router)
app.include_router(chatbot.router)

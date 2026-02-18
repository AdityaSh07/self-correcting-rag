from fastapi import FastAPI
from . import models
from .database import engine, SessionLocal
from .routers import user, auth
from .config import settings
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials = ["*"]
)

app.include_router(auth.router)
app.include_router(user.router)

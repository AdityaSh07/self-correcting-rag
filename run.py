from fastapi import FastAPI, Response, status, HTTPException, Depends
from backend.app import models
from backend.app import database
from backend.app.routers import auth, user
from fastapi.middleware.cors import CORSMiddleware

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

# 1. Define the origins that are allowed to make requests to your API
origins = [
    "http://localhost:3000",      # React/Next.js default
    "http://localhost:5173",      # Vite default
    "http://127.0.0.1:5500",      # Live Server default
]

# 2. Add the middleware to your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],               # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],               # Allows all headers
)

app.include_router(user.router)

app.include_router(auth.router)
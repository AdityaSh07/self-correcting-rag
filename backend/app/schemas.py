from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    email: EmailStr


class ChatRequest(BaseModel):
    message: str


class TokenData(BaseModel):
    id: Optional[int] = None
from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    email: EmailStr


class ChatRequest(BaseModel):
    message: str
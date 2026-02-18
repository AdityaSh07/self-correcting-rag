from fastapi import FastAPI, APIRouter, status, Depends, HTTPException
from .. import schemas,utils, models
from sqlalchemy.orm import Session
from ..database import get_db


router = APIRouter(
    prefix='/users',
    tags=['Users']
)


@router.post('/', status_code=status.HTTP_201_CREATED, response_model= schemas.UserOut)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    
    #hash pw
    hashed_pw = utils.hash(user.password)
    user.password = hashed_pw
    
    new_user = models.User(**user.model_dump())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user
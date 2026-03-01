from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from .. import database
from .. import utils, models, oauth2
from fastapi import Response
from ..config import settings

router = APIRouter(
    tags=["Login"]
)


@router.post('/login')
def login(
    response: Response,
    user_credentials: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(database.get_db)
):
    user = db.query(models.User).filter(models.User.email == user_credentials.username).first()
    
    if not user or not utils.verify(user_credentials.password, user.password):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Credentials")
    
    access_token = oauth2.create_access_token(data={'user_id': user.id})

    response.set_cookie(
        key="access_token", 
        value=access_token, 
        httponly=True,   # anti xss
        max_age=settings.access_token_expire_minutes*60,    # 24hrs
        samesite="lax",  # CSRF protection
        secure=False     # set true in production with HTTPS
    )

    return {"message": "Login successful"}



@router.post('/logout')
def logout(response: Response):
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="lax",
        secure=False
    )
    return {"message": "Successfully logged out"}
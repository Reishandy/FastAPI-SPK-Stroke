from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from schemas import UserResponse, PersonalDefaults
from security import SECRET_KEY, ALGORITHM
from database import users_collection, fix_id

router = APIRouter(prefix="/users", tags=["User Profile"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


# --- DEPENDENCY: Get Current User ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Database Lookup
    user = await users_collection.find_one({"email": email})
    if user is None:
        raise credentials_exception

    return fix_id(user)


# --- ENDPOINTS ---

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user


@router.put("/me/defaults", status_code=204)
async def update_user_defaults(defaults: PersonalDefaults, current_user: dict = Depends(get_current_user)):
    # Update only the personal_defaults field
    await users_collection.update_one(
        {"email": current_user["email"]},
        {"$set": {"personal_defaults": defaults.dict(exclude_unset=True)}}
    )
    return
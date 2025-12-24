from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from schemas import UserResponse, PersonalDefaults
from security import SECRET_KEY, ALGORITHM
from routers.auth import fake_users_db  # TODO: Import the mock DB

router = APIRouter(prefix="/users", tags=["User Profile"])

# This tells FastAPI that the token comes from the /auth/tokenUrl
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


# --- DEPENDENCY: Get Current User ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode Token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Lookup User in Mock DB
    user = fake_users_db.get(email)
    if user is None:
        raise credentials_exception
    return user


# --- ENDPOINTS ---

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """
    Returns the profile of the currently logged-in user.
    """
    return current_user


@router.put("/me/defaults", status_code=204)
async def update_user_defaults(defaults: PersonalDefaults, current_user: dict = Depends(get_current_user)):
    """
    Mock update of user defaults.
    """
    current_user["personal_defaults"] = defaults.dict(exclude_unset=True)
    return
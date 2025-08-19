import os
from fastapi_users import FastAPIUsers, BaseUserManager, IntegerIDMixin
from fastapi_users.authentication import CookieTransport, AuthenticationBackend, JWTStrategy
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from sqlalchemy import select
from db import User, AsyncSessionLocal
from fastapi import Depends  


# =============================================================================
# CONFIGURACIÓN DE AUTENTICACIÓN
# =============================================================================


cookie_transport = CookieTransport(cookie_max_age=3600)
SECRET = os.getenv("SECRET_KEY")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = os.getenv("SECRET_KEY")
    verification_token_secret = os.getenv("SECRET_KEY")

    async def authenticate(self, credentials: OAuth2PasswordRequestForm) -> User | None:
        async with AsyncSessionLocal() as session:
            user = await session.execute(select(User).where(User.username == credentials.username))
            user = user.scalars().first()
            if user and pwd_context.verify(credentials.password, user.hashed_password):
                return user
        return None

async def get_user_db():
    async with AsyncSessionLocal() as session:
        yield SQLAlchemyUserDatabase(session, User)

async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)

fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

auth_router = fastapi_users.get_auth_router(auth_backend)
current_user = fastapi_users.current_user(active=True)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
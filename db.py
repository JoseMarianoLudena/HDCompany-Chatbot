import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.sql import func
# =============================================================================
# CONFIGURACIÓN DE BASE DE DATOS
# =============================================================================
def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return database_url
    else:
        return "sqlite+aiosqlite:///hdcompany.db"

DATABASE_URL = get_database_url()

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# =============================================================================
# MODELOS DE BASE DE DATOS
# =============================================================================


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    user_phone = Column(String, ForeignKey("clients.phone"))
    name = Column(String)
    is_group = Column(String, default="False")
    group_id = Column(String, nullable=True)
    state = Column(String, default="active")
    escalated = Column(String, default="False")
    created_at = Column(DateTime, default=func.now())

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    sender = Column(String)
    message = Column(String)
    timestamp = Column(DateTime, default=func.now())

class Cart(Base):
    __tablename__ = "carts"
    id = Column(Integer, primary_key=True, index=True)
    user_phone = Column(String, ForeignKey("clients.phone"))
    product_name = Column(String)
    product_price = Column(Float)
    added_at = Column(DateTime, default=func.now())

class Client(Base):
    __tablename__ = "clients"
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String, unique=True, index=True)
    name = Column(String)
    registered_at = Column(DateTime, default=func.now())

async def init_db(pwd_context):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("Tablas creadas:", Base.metadata.tables.keys())
    async with AsyncSessionLocal() as session:
        user = await session.execute(User.__table__.select().where(User.username == "admin"))
        if not user.scalars().first():
            hashed_password = pwd_context.hash("admin123")
            session.add(User(username="admin", hashed_password=hashed_password))
            await session.commit()
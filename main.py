import json
import os
from datetime import datetime, timezone
from fastapi import FastAPI, Request, Response, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi_users import FastAPIUsers, BaseUserManager, IntegerIDMixin
from fastapi_users.authentication import CookieTransport, AuthenticationBackend, JWTStrategy
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from crewai import Agent, Task, Crew
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy import create_engine # noqa
import socketio
import aiosqlite # noqa
from passlib.context import CryptContext
from contextlib import asynccontextmanager
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import Union
import re
import asyncio
import unicodedata

# Cargar variables de entorno
load_dotenv()

# Inicializar la aplicación FastAPI
app = FastAPI()
# Expone la carpeta /images como accesible públicamente en la URL /images
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

# Configurar Socket.IO
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app.mount("/socket.io", socketio.ASGIApp(sio))

# Usar DATABASE_URL de entorno (PostgreSQL en Render, SQLite localmente si no está definida)
def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        # En Render, cambiar postgresql:// por postgresql+asyncpg://
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return database_url
    else:
        # Desarrollo local con SQLite
        return "sqlite+aiosqlite:///hdcompany.db"

DATABASE_URL = get_database_url()

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Definir la base para los modelos de SQLAlchemy
Base = declarative_base()

# Modelo de usuario para autenticación
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)

# Modelo para conversaciones
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

# Modelo para mensajes
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    sender = Column(String)
    message = Column(String)
    timestamp = Column(DateTime, default=func.now())

# Modelo para Carrito de Compras
class Cart(Base):
    __tablename__ = "carts"
    id = Column(Integer, primary_key=True, index=True)
    user_phone = Column(String, ForeignKey("clients.phone"))
    product_name = Column(String)
    product_price = Column(Float)
    added_at = Column(DateTime, default=func.now())

# Modelo para clientes
class Client(Base):
    __tablename__ = "clients"
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String, unique=True, index=True)
    name = Column(String)
    registered_at = Column(DateTime, default=func.now())

# Configurar autenticación
cookie_transport = CookieTransport(cookie_max_age=3600)
SECRET = os.getenv("SECRET_KEY")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

# Definir UserManager
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

async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(lambda: SQLAlchemyUserDatabase(User, AsyncSessionLocal))):
    yield UserManager(user_db)

fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

# Incluir el enrutador de autenticación
auth_router = fastapi_users.get_auth_router(auth_backend)
app.include_router(auth_router, prefix="/auth")

current_user = fastapi_users.current_user(active=True)

# Configurar passlib para hash de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Cargar productos desde JSON
try:
    with open("data/products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
except FileNotFoundError:
    print("Error: products.json no encontrado. Creando una lista vacía.")
    products = []
except json.JSONDecodeError as e:
    print(f"Error al decodificar JSON: {e}. Creando una lista vacía.")
    products = []

# Cargar FAQs desde JSON
try:
    with open("data/faqs.json", "r", encoding="utf-8") as f:
        faqs = json.load(f)
except FileNotFoundError:
    print("Error: faqs.json no encontrado. Creando una lista vacía.")
    faqs = []
except json.JSONDecodeError as e:
    print(f"Error al decodificar faqs.json: {e}. Creando una lista vacía.")
    faqs = []

def normalize_text(text: str) -> str:
    """Normaliza texto para comparación insensible a mayúsculas, tildes y comillas."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.replace("″", '"').replace(""", '"').replace(""", '"')
    return text.strip().lower()

def match_product_name(cart_name: str, product_name: str) -> bool:
    """Coincide productos aunque los nombres no sean exactos, usando palabras clave."""
    cart_name_norm = normalize_text(cart_name)
    product_name_norm = normalize_text(product_name)
    cart_words = set(cart_name_norm.split())
    product_words = set(product_name_norm.split())
    return len(cart_words & product_words) >= 3  # al menos 3 palabras en común

# Función para buscar productos por categoría o palabra clave
def search_products(query: str, category: str = None):
    """Busca productos en el inventario basado en la consulta"""
    query_lower = query.lower()
    found_products = []
    for product in products:
        product_name_lower = product.get("nombre", "").lower()
        if category and category.lower() not in product.get("categoria", "").lower():
            continue
        # Filtrar estrictamente por nombres que empiecen con "laptop" o "impresora"
        if query_lower == "laptops" and product_name_lower.startswith(("laptop ", "laptops ")) and "laptops" in product.get("categoria", "").lower():
            found_products.append(product)
        elif query_lower == "impresoras" and product_name_lower.startswith(("impresora ", "impresoras ")) and "impresoras" in product.get("categoria", "").lower():
            found_products.append(product)
        elif query_lower == "accesorios" and "accesorios" in product.get("categoria", "").lower():
            found_products.append(product)
        elif query_lower == "ofertas" and "oferta" in product.get("descripcion", "").lower():
            found_products.append(product)
        elif query_lower not in ["laptops", "impresoras", "accesorios", "ofertas"] and (
            query_lower in product_name_lower or 
            query_lower in product.get("descripcion", "").lower() or 
            query_lower in product.get("categoria", "").lower()
        ):
            found_products.append(product)
    return found_products

def format_products_response(products_list, query=""):
    """Formatea la respuesta con los productos encontrados"""
    if not products_list:
        return f"Lo siento, no encontré productos relacionados con '{query}' en nuestro inventario actual."
    response = f"Estos son los productos que tenemos relacionados con '{query}':\n\n"
    for i, product in enumerate(products_list[:5], 1):
        response += f"{i}. **{product['nombre']}**\n"
        response += f"   💰 Precio: {product['precio']}\n"
        response += f"   📝 {product['descripcion']}\n"
        if product.get('categoria'):
            response += f"   🏷️ Categoría: {product['categoria']}\n"
        response += "\n"
    if len(products_list) > 5:
        response += f"Y {len(products_list) - 5} productos más disponibles.\n"
    response += "¿Te interesa alguno en particular? ¿Necesitas más información?"
    return response

def create_system_prompt():
    """Crea el prompt del sistema con información de productos y FAQs"""
    categories = set()
    product_names = []
    for product in products:
        if product.get('categoria'):
            categories.add(product['categoria'])
        product_names.append(product['nombre'])
    categories_str = ", ".join(sorted(categories))
    faqs_str = "\n".join([f"Pregunta: {faq['question']}\nRespuesta: {faq['answer']}" for faq in faqs])
    with open("data/system_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template.format(
        len_products=len(products),
        categories_str=categories_str,
        faqs_str=faqs_str
    )

# Inicializar LangChain LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.7
)

# Diccionario para almacenar historiales por usuario
user_chat_histories = {}

def get_chat_history(user_phone: str):
    """Obtiene o crea el historial de chat para un usuario específico"""
    if user_phone not in user_chat_histories:
        user_chat_histories[user_phone] = InMemoryChatMessageHistory()
    return user_chat_histories[user_phone]

# Inicialización --> memory
chat_history = InMemoryChatMessageHistory()
conversation = RunnableWithMessageHistory(
    llm,
    lambda: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="response"
)

# Definir agente CrewAI para consultas de productos
sales_agent = Agent(
    role="Asistente de Ventas",
    goal="Asistir a los clientes con consultas de productos y recomendaciones",
    backstory="Eres un asistente de ventas experto en HD Company, especializado en laptops, impresoras y accesorios tecnológicos.",
    llm=llm,
    verbose=True
)

# Definir tarea CrewAI para recomendación de productos
recommend_task = Task(
    description="Basado en la entrada del usuario, recomienda 2-3 productos de la lista proporcionada. Incluye nombre, precio y descripción.",
    agent=sales_agent,
    expected_output="Una lista de 2-3 productos recomendados con sus nombres, precios y descripciones."
)

# Crew para manejar tareas
sales_crew = Crew(
    agents=[sales_agent],
    tasks=[recommend_task],
    verbose=True
)

# Modelo Pydantic para mensajes de WhatsApp - ACTUALIZADO PARA BOTONES
class WhatsAppMessage(BaseModel):
    text: str | None = None
    from_: str | None = Field(None, alias="from")
    message_type: str = "text"
    from_number: str | None = None
    message_body: str | None = None
    button_id: str | None = None
    list_reply: dict | None = None
    # NUEVOS CAMPOS PARA INTERACTIVOS
    button_reply: dict | None = None
    interactive_reply: dict | None = None

    def get_message_content(self) -> str | None:
        # Manejar respuestas de botones
        if self.button_reply:
            return self.button_reply.get("id")
        # Manejar respuestas de listas
        if self.list_reply:
            return self.list_reply.get("id")
        # Manejar respuestas interactivas generales
        if self.interactive_reply:
            if self.interactive_reply.get("type") == "button_reply":
                return self.interactive_reply.get("button_reply", {}).get("id")
            elif self.interactive_reply.get("type") == "list_reply":
                return self.interactive_reply.get("list_reply", {}).get("id")
        return self.message_body or self.text
    
    def get_from_number(self) -> str:
        # Asegura que siempre devuelve un string, usando from_ si from_number está vacío
        return (self.from_number or getattr(self, "from_", None) or "").strip()

# Modelo Pydantic para mensajes del dashboard
class DashboardMessage(BaseModel):
    user_phone: str
    message: str

# NUEVAS FUNCIONES PARA CREAR MENSAJES INTERACTIVOS
def create_main_menu():
    """Crea el menú principal con lista de opciones"""
    return {
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {
                "text": "¡Bienvenido a HD Company! 🏪\nSomos tu tienda de confianza para laptops, impresoras y accesorios tecnológicos.\n\n¿En qué te puedo ayudar hoy?"
            },
            "action": {
                "button": "Ver Opciones 📋",
                "sections": [
                    {
                        "title": "Menú Principal",
                        "rows": [
                            {"id": "ofertas", "title": "🔥 Ofertas", "description": "Ver productos en promoción"},
                            {"id": "laptops", "title": "💻 Laptops", "description": "Explorar nuestras laptops"},
                            {"id": "impresoras", "title": "🖨️ Impresoras", "description": "Ver impresoras disponibles"},
                            {"id": "accesorios", "title": "🖱️ Accesorios", "description": "Accesorios y componentes"},
                            {"id": "soporte", "title": "🛠️ Soporte", "description": "Agendar soporte técnico"},
                            {"id": "agente", "title": "👤 Hablar con Agente", "description": "Conectar con un humano"}
                        ]
                    }
                ]
            }
        }
    }

def create_product_buttons():
    """Crea botones para acciones de productos"""
    return {
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": "¿Qué te gustaría hacer con este producto?"
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "add_to_cart",
                            "title": "🛒 Agregar al Carrito"
                        }
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "view_image",
                            "title": "🖼️ Ver Imagen"
                        }
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "continue_browsing",
                            "title": "👀 Seguir Viendo"
                        }
                    }
                ]
            }
        }
    }

def create_cart_buttons():
    """Crea botones para acciones del carrito"""
    return {
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": "¿Qué quieres hacer con tu carrito?"
            },
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {
                            "id": "view_cart",
                            "title": "👀 Ver Carrito"
                        }
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "checkout",
                            "title": "💳 Finalizar Compra"
                        }
                    },
                    {
                        "type": "reply",
                        "reply": {
                            "id": "continue_shopping",
                            "title": "🛍️ Seguir Comprando"
                        }
                    }
                ]
            }
        }
    }

def create_accessories_menu():
    """Crea menú de categorías de accesorios"""
    return {
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {
                "text": "📋 Selecciona la categoría de accesorios que te interesa:"
            },
            "action": {
                "button": "Ver Categorías 📂",
                "sections": [
                    {
                        "title": "Categorías de Accesorios",
                        "rows": [
                            {"id": "case", "title": "📱 Cases", "description": "Fundas y protectores"},
                            {"id": "camaras", "title": "📷 Cámaras", "description": "Cámaras y webcams"},
                            {"id": "discos", "title": "💾 Discos", "description": "Almacenamiento externo"},
                            {"id": "monitores", "title": "🖥️ Monitores", "description": "Pantallas y monitores"},
                            {"id": "mouse_teclado", "title": "⌨️ Mouse y Teclado", "description": "Periféricos de entrada"},
                            {"id": "tarjetas_video", "title": "🎮 Tarjetas de Video", "description": "GPUs y tarjetas gráficas"},
                            {"id": "tablets", "title": "📱 Tablets", "description": "Tablets y accesorios"}
                        ]
                    }
                ]
            }
        }
    }

# Crear tablas en la base de datos y usuario predeterminado
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # ✅ Crea sin borrar
        print("Tablas creadas:", Base.metadata.tables.keys())
    async with AsyncSessionLocal() as session:
        user = await session.execute(User.__table__.select().where(User.username == "admin"))
        if not user.scalars().first():
            hashed_password = pwd_context.hash("admin123")
            session.add(User(username="admin", hashed_password=hashed_password))
            await session.commit()

# Socket.IO eventos
@sio.event
async def connect(sid, environ):
    print(f"Cliente conectado: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Cliente desconectado: {sid}")

# Ruta para login
@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Ruta para login
@app.post("/login")
async def post_login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_manager: UserManager = Depends(get_user_manager)
):
    user = await user_manager.authenticate(form_data)
    if not user:
        return templates.TemplateResponse("login.html", {
            "request": Request,
            "messages": [("error", "Usuario o contraseña incorrectos")]
        })
    
    # Generar el token JWT y configurar la cookie
    jwt_strategy = get_jwt_strategy()
    token = await jwt_strategy.create_access_token(data={"sub": str(user.id)})
    cookie_transport.write(response, token)
    
    return RedirectResponse(url="/dashboard", status_code=303)

# Ruta para logout
@app.get("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(cookie_transport.cookie_name)
    return response

# Ruta para dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request, user: User = Depends(current_user)):
    async with AsyncSessionLocal() as session:
        conversations = {}
        convs = await session.execute(Conversation.__table__.select())
        for conv in convs.scalars().all():
            messages = await session.execute(
                Message.__table__.select().where(Message.conversation_id == conv.id)
            )
            conversations[conv.user_phone] = {
                "name": conv.name,
                "is_group": conv.is_group,
                "group_id": conv.group_id,
                "state": conv.state,
                "escalated": conv.escalated,
                "messages": [
                    {"sender": msg.sender, "message": msg.message, "timestamp": msg.timestamp.isoformat()}
                    for msg in messages.scalars().all()
                ],
                "active_poll": None
            }
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "current_user": user,
            "conversations": conversations,
            "url_for": lambda x: f"/{x}"
        })

# Ruta para clientes
@app.get("/clients", response_class=HTMLResponse)
async def get_clients(request: Request, user: User = Depends(current_user)):
    async with AsyncSessionLocal() as session:
        clients = await session.execute(Client.__table__.select())
        clients = [(c.phone, c.name, c.registered_at.isoformat()) for c in clients.scalars().all()]
        return templates.TemplateResponse("clients.html", {
            "request": request,
            "current_user": user,
            "clients": clients,
            "url_for": lambda x: f"/{x}"
        })

# Endpoint para mensajes del dashboard
@app.post("/dashboard")
async def post_dashboard(message: DashboardMessage, user: User = Depends(current_user)):
    async with AsyncSessionLocal() as session:
        conv = await session.execute(
            Conversation.__table__.select().where(Conversation.user_phone == message.user_phone)
        )
        conv = conv.scalars().first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversación no encontrada")
        new_message = Message(
            conversation_id=conv.id,
            sender="agent",
            message=message.message,
            timestamp=datetime.utcnow()
        )
        session.add(new_message)
        await session.commit()
        await sio.emit("new_message", {
            "user_phone": message.user_phone,
            "name": conv.name,
            "is_group": conv.is_group,
            "group_id": conv.group_id,
            "state": conv.state,
            "escalated": conv.escalated,
            "message": message.message,
            "sender": "agent",
            "timestamp": datetime.utcnow().isoformat(),
            "active_poll": None
        })
        return {"status": "Mensaje enviado"}

# Endpoint webhook para mensajes de WhatsApp - ACTUALIZADO CON BOTONES
@app.post("/process")
async def process_message(message: WhatsAppMessage):
    try:
        print(f"Datos recibidos: {message.model_dump()}")
        user_input = message.get_message_content() if message.message_type in ["text", "button", "interactive"] else ""
        from_number = message.get_from_number()
        print(f"Mensaje: {user_input}")
        print(f"Número: {from_number}")
        
        if not from_number:
            raise HTTPException(status_code=400, detail="Número de teléfono requerido")

        async with AsyncSessionLocal() as session:
            # Obtener o crear cliente
            result = await session.execute(select(Client).where(Client.phone == from_number))
            client = result.scalars().first()
            if not client:
                client = Client(phone=from_number, name="Desconocido")
                session.add(client)
                await session.commit()
                await session.refresh(client)

            # Obtener o crear conversación
            result = await session.execute(select(Conversation).where(Conversation.user_phone == from_number))
            conv = result.scalars().first()
            if not conv:
                conv = Conversation(user_phone=from_number, name=client.name, escalated="False", state="active")
                session.add(conv)
                await session.commit()
                await session.refresh(conv)

            # Registrar mensaje del usuario
            if user_input:
                new_message = Message(
                    conversation_id=conv.id,
                    sender="client",
                    message=user_input,
                    timestamp=datetime.utcnow()
                )
                session.add(new_message)
                await session.commit()
                await sio.emit("new_message", {
                    "user_phone": from_number,
                    "name": conv.name,
                    "is_group": conv.is_group,
                    "group_id": conv.group_id,
                    "state": conv.state,
                    "escalated": conv.escalated,
                    "message": user_input,
                    "sender": "client",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_poll": None
                })

            # MANEJAR ESTADO INICIAL (pedir nombre) - CON MENÚ PRINCIPAL
            if client.name == "Desconocido":
                if not user_input or user_input.lower() in ["hola", "hi", "hey", "buenas", "hello"]:
                    response_body = {
                        "type": "text",
                        "body": "😊 ¡Hola! Soy el asistente de HD Company. ¿Cuál es tu nombre?"
                    }
                else:
                    client.name = user_input.strip()
                    conv.name = user_input.strip()
                    session.add(client)
                    session.add(conv)
                    await session.commit()
                    
                    # ENVIAR MENÚ PRINCIPAL DESPUÉS DE OBTENER EL NOMBRE
                    response_body = create_main_menu()

                # Registrar respuesta
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=json.dumps(response_body) if isinstance(response_body, dict) else response_body,
                    timestamp=datetime.utcnow()
                )
                session.add(bot_message)
                await session.commit()
                await sio.emit("new_message", {
                    "user_phone": from_number,
                    "name": conv.name,
                    "is_group": conv.is_group,
                    "group_id": conv.group_id,
                    "state": conv.state,
                    "escalated": conv.escalated,
                    "message": json.dumps(response_body) if isinstance(response_body, dict) else response_body,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_poll": None
                })
                return response_body

            # MANEJAR ESTADO ESCALADO
            if conv.escalated == "True":
                if user_input.lower() == "volver":
                    conv.escalated = "False"
                    conv.state = "active"
                    await session.commit()
                    response_body = create_main_menu()
                else:
                    response_body = {
                        "type": "text",
                        "body": "🔔 Estás conectado con un agente. Escribe 'volver' para regresar al menú."
                    }
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=json.dumps(response_body) if isinstance(response_body, dict) else response_body,
                    timestamp=datetime.utcnow()
                )
                session.add(bot_message)
                await session.commit()
                await sio.emit("new_message", {
                    "user_phone": from_number,
                    "name": conv.name,
                    "is_group": conv.is_group,
                    "group_id": conv.group_id,
                    "state": conv.state,
                    "escalated": conv.escalated,
                    "message": json.dumps(response_body) if isinstance(response_body, dict) else response_body,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_poll": None
                })
                return response_body

            # MANEJAR CIERRE DE CONVERSACIÓN
            if user_input.lower() in ["gracias", "resuelto", "listo", "ok", "solucionado"]:
                response_body = {
                    "type": "text",
                    "body": "¡Gracias por contactarnos! 😊 Escríbenos si necesitas más ayuda."
                }
                conv.state = "closed"
                await session.commit()
                await sio.emit("close_conversation", {"user_phone": from_number}, namespace="/dashboard")
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=json.dumps(response_body),
                    timestamp=datetime.utcnow()
                )
                session.add(bot_message)
                await session.commit()
                await sio.emit("new_message", {
                    "user_phone": from_number,
                    "name": conv.name,
                    "is_group": conv.is_group,
                    "group_id": conv.group_id,
                    "state": conv.state,
                    "escalated": conv.escalated,
                    "message": json.dumps(response_body),
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_poll": None
                })
                return response_body

            # MANEJAR RESPUESTAS DE BOTONES Y LISTAS
            if user_input:
                # Respuestas del menú principal
                if user_input == "ofertas":
                    products_list = search_products("ofertas")
                    response_text = format_products_response(products_list, "ofertas")
                    response_body = {
                        "type": "text",
                        "body": response_text
                    }
                
                elif user_input == "laptops":
                    products_list = search_products("laptops")
                    response_text = format_products_response(products_list, "laptops")
                    response_body = {
                        "type": "text",
                        "body": response_text
                    }
                
                elif user_input == "impresoras":
                    products_list = search_products("impresoras")
                    response_text = format_products_response(products_list, "impresoras")
                    response_body = {
                        "type": "text",
                        "body": response_text
                    }
                
                elif user_input == "accesorios":
                    response_body = create_accessories_menu()
                
                elif user_input == "soporte":
                    response_body = {
                        "type": "text",
                        "body": f"📅 Agendar soporte técnico: https://calendly.com/hdcompany/soporte\n\n¿En qué más te puedo ayudar, {conv.name}?"
                    }
                
                elif user_input == "agente":
                    conv.escalated = "True"
                    conv.state = "escalated"
                    await session.commit()
                    response_body = {
                        "type": "text",
                        "body": "🔔 Te conecto con un agente. ¡Un momento! 😊"
                    }
                    # Notificar al agente (opcional - requiere configuración adicional)
                    # send_agent_notification(from_number, conv.name)
                
                # Respuestas de categorías de accesorios
                elif user_input in ["case", "camaras", "discos", "monitores", "mouse_teclado", "tarjetas_video", "tablets"]:
                    products_list = search_products(user_input)
                    response_text = format_products_response(products_list, user_input)
                    response_body = {
                        "type": "text",
                        "body": response_text
                    }
                
                # Respuestas de botones de producto
                elif user_input == "add_to_cart":
                    # Buscar último producto mencionado y agregarlo al carrito
                    previous_messages = await session.execute(
                        select(Message).where(Message.conversation_id == conv.id).order_by(Message.timestamp.desc()).limit(10)
                    )
                    previous_messages = previous_messages.scalars().all()
                    
                    last_selected = None
                    for msg in reversed(previous_messages):
                        if msg.sender == 'agent' and isinstance(msg.message, str):
                            m = re.search(r'\*\*(.+?)\*\*', msg.message)
                            if m:
                                last_selected = m.group(1).strip()
                                break
                    
                    if last_selected:
                        # Buscar producto en la lista
                        product_found = None
                        for product in products:
                            if product['nombre'] == last_selected:
                                product_found = product
                                break
                        
                        if product_found:
                            cart_item = Cart(
                                user_phone=from_number,
                                product_name=product_found['nombre'],
                                product_price=float(product_found['precio'].replace("PEN ", "")),
                                added_at=datetime.utcnow()
                            )
                            session.add(cart_item)
                            await session.commit()
                            response_body = create_cart_buttons()
                        else:
                            response_body = {
                                "type": "text",
                                "body": "No pude encontrar el producto para agregar al carrito. Por favor selecciona un producto primero."
                            }
                    else:
                        response_body = {
                            "type": "text",
                            "body": "No hay producto seleccionado. Por favor selecciona un producto primero."
                        }
                
                elif user_input == "view_image":
                    # Buscar imagen del último producto mencionado
                    previous_messages = await session.execute(
                        select(Message).where(Message.conversation_id == conv.id).order_by(Message.timestamp.desc()).limit(10)
                    )
                    previous_messages = previous_messages.scalars().all()
                    
                    last_selected = None
                    for msg in reversed(previous_messages):
                        if msg.sender == 'agent' and isinstance(msg.message, str):
                            m = re.search(r'\*\*(.+?)\*\*', msg.message)
                            if m:
                                last_selected = m.group(1).strip()
                                break
                    
                    if last_selected:
                        # Buscar imagen del producto
                        base_url = "https://hdcompany-chatbot.onrender.com"
                        product_found = None
                        for product in products:
                            if product['nombre'] == last_selected:
                                product_found = product
                                break
                        
                        if product_found and product_found.get('image_url'):
                            img_url = product_found['image_url']
                            if not img_url.startswith('http'):
                                if not img_url.startswith('/'):
                                    img_url = '/' + img_url
                                img_url = f"{base_url}{img_url}"
                            
                            response_body = {
                                "type": "image",
                                "image": {"link": img_url},
                                "caption": f"🖼️ Imagen de {product_found['nombre']}\n\n¿Qué más te gustaría hacer?"
                            }
                        else:
                            response_body = {
                                "type": "text",
                                "body": "Lo siento, no hay imagen disponible para este producto."
                            }
                    else:
                        response_body = {
                            "type": "text",
                            "body": "No hay producto seleccionado. Por favor selecciona un producto primero."
                        }
                
                elif user_input == "continue_browsing":
                    response_body = create_main_menu()
                
                elif user_input == "view_cart":
                    # Mostrar contenido del carrito
                    cart_items = await session.execute(
                        select(Cart).where(Cart.user_phone == from_number)
                    )
                    cart_items = cart_items.scalars().all()
                    
                    if cart_items:
                        cart_text = f"🛒 Tu carrito ({len(cart_items)} productos):\n\n"
                        total = 0
                        for i, item in enumerate(cart_items, 1):
                            cart_text += f"{i}. {item.product_name}\n"
                            cart_text += f"   💰 PEN {item.product_price}\n\n"
                            total += item.product_price
                        cart_text += f"💰 **Total: PEN {total:.2f}**"
                        response_body = {
                            "type": "text",
                            "body": cart_text
                        }
                    else:
                        response_body = {
                            "type": "text",
                            "body": "🛒 Tu carrito está vacío.\n\n¿Te gustaría ver nuestros productos?"
                        }
                
                elif user_input == "checkout":
                    response_body = {
                        "type": "text",
                        "body": "💳 Para finalizar tu compra, por favor contacta con nuestro agente de ventas.\n\n¿Te conecto con un agente?"
                    }
                
                elif user_input == "continue_shopping":
                    response_body = create_main_menu()
                
                # Si no es una respuesta de botón/lista reconocida, usar LLM
                else:
                    # Aquí mantener la lógica original del LLM para consultas genéricas
                    try:
                        # Obtener contexto de conversación previa
                        previous_messages = await session.execute(
                            select(Message).where(Message.conversation_id == conv.id).order_by(Message.timestamp.desc()).limit(10)
                        )
                        previous_messages = previous_messages.scalars().all()

                        conversation_context = "Historial de conversación reciente:\n"
                        for msg in reversed(previous_messages[1:]):
                            conversation_context += f"{msg.sender}: {msg.message}\n"

                        # Ajustar URLs relativas a absolutas
                        base_url = "https://hdcompany-chatbot.onrender.com"
                        products_with_absolute_urls = []
                        for product in products:
                            img_url = product.get('image_url', '')
                            if img_url and not img_url.startswith('http'):
                                if not img_url.startswith('/'):
                                    img_url = '/' + img_url
                                img_url = f"{base_url}{img_url}"
                            products_with_absolute_urls.append({**product, "image_url": img_url or product.get('image_url', '')})

                        products_context = f"Productos disponibles: {json.dumps(products_with_absolute_urls, ensure_ascii=False)}"

                        # Construir contexto del carrito
                        cart_items = await session.execute(
                            select(Cart).where(Cart.user_phone == from_number)
                        )
                        cart_items = cart_items.scalars().all()

                        if cart_items:
                            cart_lines = []
                            for item in cart_items:
                                imagen_url = next(
                                    (
                                        p['image_url']
                                        for p in products_with_absolute_urls
                                        if match_product_name(item.product_name, p['nombre'])
                                    ),
                                    'No disponible'
                                )
                                cart_lines.append(f"{item.product_name}: PEN {item.product_price}, Imagen: {imagen_url}")
                            cart_context = "Carrito actual del usuario:\n" + "\n".join(cart_lines)
                        else:
                            cart_context = "Carrito vacío"

                        messages = [
                            {"role": "system", "content": create_system_prompt()},
                            {"role": "user", "content": f"Contexto de productos: {products_context}\n\nContexto del carrito: {cart_context}\n\n{conversation_context}\n\nConsulta actual del cliente: {user_input}"}
                        ]
                        response = await llm.ainvoke(messages)
                        response_text = response.content

                        # Detectar si la respuesta es una imagen
                        image_url_match = re.match(r'.*?(https?://[^\s]+(?:\.png|\.jpg|\.jpeg|\.gif))', response_text)
                        
                        if image_url_match:
                            image_url = image_url_match.group(1).strip()
                            response_body = {
                                "type": "image",
                                "image": {"link": image_url},
                                "caption": f"🖼️ Imagen solicitada. ¿En qué más te ayudo, {conv.name}?"
                            }
                        else:
                            # Agregar botones de producto si la respuesta menciona un producto específico
                            if "has seleccionado" in response_text or re.search(r'\*\*(.+?)\*\*', response_text):
                                response_body = [
                                    {
                                        "type": "text",
                                        "body": response_text
                                    },
                                    create_product_buttons()
                                ]
                            else:
                                response_body = {
                                    "type": "text",
                                    "body": response_text
                                }

                        # Procesar respuesta del LLM para detectar acciones como agregar al carrito
                        if "Producto agregado al carrito" in response_text:
                            product_name_match = re.search(r"has seleccionado (.+?)\.", response_text)
                            if product_name_match:
                                product_name = product_name_match.group(1)
                                for product in products:
                                    if product['nombre'] == product_name:
                                        cart_item = Cart(
                                            user_phone=from_number,
                                            product_name=product['nombre'],
                                            product_price=float(product['precio'].replace("PEN ", "")),
                                            added_at=datetime.utcnow()
                                        )
                                        session.add(cart_item)
                                        await session.commit()
                                        break
                    except Exception as e:
                        print(f"Error con LLM: {e}")
                        response_body = {
                            "type": "text",
                            "body": f"🛠️ Lo siento, estamos en mantenimiento. Por favor, intenta de nuevo más tarde. ¿En qué te ayudo ahora, {conv.name}?"
                        }

            # Si response_body es una lista, enviar múltiples mensajes
            if isinstance(response_body, list):
                for response_item in response_body:
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=json.dumps(response_item),
                        timestamp=datetime.utcnow()
                    )
                    session.add(bot_message)
                    await session.commit()
                    await sio.emit("new_message", {
                        "user_phone": from_number,
                        "name": conv.name,
                        "is_group": conv.is_group,
                        "group_id": conv.group_id,
                        "state": conv.state,
                        "escalated": conv.escalated,
                        "message": json.dumps(response_item),
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_poll": None
                    })
                return response_body[-1]  # Retornar el último mensaje
            else:
                # Mensaje único
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=json.dumps(response_body) if isinstance(response_body, dict) else response_body,
                    timestamp=datetime.utcnow()
                )
                session.add(bot_message)
                await session.commit()
                await sio.emit("new_message", {
                    "user_phone": from_number,
                    "name": conv.name,
                    "is_group": conv.is_group,
                    "group_id": conv.group_id,
                    "state": conv.state,
                    "escalated": conv.escalated,
                    "message": json.dumps(response_body) if isinstance(response_body, dict) else response_body,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_poll": None
                })
                return response_body

    except Exception as e:
        print(f"Error en process_message: {e}")
        import traceback
        traceback.print_exc()
        return {
            "type": "text",
            "body": "Lo siento, hubo un error procesando tu mensaje. Por favor intenta de nuevo."
        }

# Ruta raíz para redirigir a login
@app.get("/")
async def root():
    return RedirectResponse(url="/login", status_code=303)

@app.on_event("startup")
async def run_init_db():
    print("🔧 Ejecutando init_db() desde @app.on_event startup")
    await init_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
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
from sqlalchemy import delete
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
import requests

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN INICIAL
# =============================================================================

greetings = ["hola", "hi", "hey", "buenas", "hello", "hola de nuevo", "hola denuevo", "hola denuevo bro", "buen dia", "buenas tardes", "buenas noches"]

main_list_options = [
    {"id": "ofertas", "title": "🔥 Ofertas Especiales"},
    {"id": "laptops", "title": "💻 Laptops"},
    {"id": "impresoras", "title": "🖨️ Impresoras"},
    {"id": "accesorios", "title": "🖱️ Accesorios"},
    {"id": "soporte", "title": "🛠️ Soporte Técnico"},
    {"id": "agente", "title": "👤 Hablar con Agente"}
]

# Cargar variables de entorno
load_dotenv()

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def remove_emojis(text: str) -> str:
    """Elimina emojis de un texto"""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticonos
        "\U0001F300-\U0001F5FF"  # Símbolos y pictogramas
        "\U0001F680-\U0001F6FF"  # Transporte y mapas
        "\U0001F700-\U0001F77F"  # Símbolos alquímicos
        "\U0001F780-\U0001F7FF"  # Símbolos geométricos
        "\U0001F800-\U0001F8FF"  # Símbolos suplementarios
        "\U0001F900-\U0001F9FF"  # Emoticonos suplementarios
        "\U0001FA00-\U0001FA6F"  # Símbolos de ajedrez
        "\U0001FA70-\U0001FAFF"  # Símbolos suplementarios
        "\U00002700-\U000027BF"  # Símbolos dingbats
        "\U00002600-\U000026FF"  # Símbolos misceláneos
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r"", text).strip()

def normalize_text(text: str) -> str:
    """Normaliza texto para comparación insensible a mayúsculas, tildes y comillas."""
    if not text:
        return ""
    
    # NORMALIZACIÓN COMPLETA - QUITA TILDES
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))  # ← LÍNEA AGREGADA
    
    # Reemplazar comillas especiales
    text = text.replace("″", '"').replace(""", '"').replace(""", '"')
    
    return text.strip().lower()

def match_product_name(cart_name: str, product_name: str) -> bool:
    """Coincide productos aunque los nombres no sean exactos, usando palabras clave."""
    cart_name_norm = normalize_text(cart_name)
    product_name_norm = normalize_text(product_name)
    cart_words = set(cart_name_norm.split())
    product_words = set(product_name_norm.split())
    return len(cart_words & product_words) >= 3  # al menos 3 palabras en común

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

# =============================================================================
# FUNCIONES DE WHATSAPP
# =============================================================================

def send_whatsapp_list(to_phone: str, message: str, list_options: list):
    """Envía una lista interactiva de WhatsApp"""
    endpoint = f"https://graph.facebook.com/v20.0/{os.getenv('WHATSAPP_PHONE_NUMBER_ID')}/messages"
    headers = {
        "Authorization": f"Bearer {os.getenv('WHATSAPP_ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone.replace("whatsapp:", ""),
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": message},
            "action": {
                "button": "Ver Opciones",
                "sections": [
                    {
                        "title": "Opciones Disponibles",
                        "rows": list_options
                    }
                ]
            }
        }
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        print(f"📋 Lista enviada: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error enviando lista: {e}")
        return False

def send_whatsapp_buttons(to_phone: str, message: str, buttons: list):
    """Envía botones interactivos de WhatsApp (máximo 3)"""
    if len(buttons) > 3:
        buttons = buttons[:3]
        
    endpoint = f"https://graph.facebook.com/v20.0/{os.getenv('WHATSAPP_PHONE_NUMBER_ID')}/messages"
    headers = {
        "Authorization": f"Bearer {os.getenv('WHATSAPP_ACCESS_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone.replace("whatsapp:", ""),
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": message},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": btn["id"], "title": btn["title"]}} 
                    for btn in buttons
                ]
            }
        }
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        print(f"🔘 Botones enviados: {response.status_code} - {response.text}")
        if response.status_code != 200:
            print(f"❌ Error en respuesta de API WhatsApp: {response.status_code} - {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error enviando botones: {e} - Payload: {json.dumps(payload, ensure_ascii=False)}")
        return False

# =============================================================================
# FUNCIONES UNIFICADAS PARA CARRITO (NUEVA ORGANIZACIÓN)
# =============================================================================

async def find_last_selected_product(session, conv_id, products):
    """Busca el último producto seleccionado en los mensajes de la conversación"""
    messages = await session.execute(
        select(Message).where(Message.conversation_id == conv_id)
        .order_by(Message.timestamp.desc()).limit(10)
    )
    messages = messages.scalars().all()
    
    for msg in messages:
        if msg.sender == 'agent' and isinstance(msg.message, str):
            # Buscar patrón específico de botones enviados
            if "Botones de producto enviados para **" in msg.message:
                # Extraer el nombre del producto del mensaje
                match = re.search(r'Botones de producto enviados para \*\*(.+?)\*\*', msg.message)
                if match:
                    product_name = match.group(1).strip()
                    print(f"🔍 Producto encontrado en mensaje: '{product_name}'")
                    
                    # Buscar en la lista de productos
                    for product in products:
                        if normalize_text(product_name) == normalize_text(product['nombre']):
                            print(f"✅ Producto coincidente encontrado: '{product['nombre']}'")
                            return product
    
    print("❌ No se encontró producto seleccionado")
    return None

async def add_product_to_cart(session, user_phone, product):
    """Función unificada para agregar productos al carrito"""
    if not product:
        return False, "No se pudo identificar el producto a agregar."
    
    try:
        # Agregar al carrito
        cart_item = Cart(
            user_phone=user_phone,
            product_name=product['nombre'],
            product_price=float(product['precio'].replace("PEN ", "")),
            added_at=datetime.utcnow()
        )
        session.add(cart_item)
        await session.commit()
        return True, f"✅ **{product['nombre']}** agregado al carrito exitosamente."
    except Exception as e:
        print(f"Error agregando al carrito: {e}")
        return False, "Error al agregar el producto al carrito."

async def get_cart_summary(session, user_phone):
    """Obtiene el resumen actual del carrito"""
    cart_items = await session.execute(
        select(Cart).where(Cart.user_phone == user_phone)
    )
    cart_items = cart_items.scalars().all()
    
    if not cart_items:
        return "🛒 Tu carrito está vacío.", 0
    
    cart_message = "🛒 **Productos del carrito:**\n"
    total_price = 0
    for i, item in enumerate(cart_items, 1):
        cart_message += f"{i}) {item.product_name}\n   💰 Precio: PEN {item.product_price}\n"
        total_price += item.product_price
    
    cart_message += f"\n💳 **Total: PEN {total_price:.2f}**"
    return cart_message, total_price

def send_cart_confirmation_buttons(from_number, cart_message):
    """Envía los botones de confirmación después de agregar al carrito"""
    cart_buttons = [
        {"id": "pagar", "title": "💳 Pagar"},
        {"id": "continue_shopping", "title": "🛍️ Seguir Viendo"},
        {"id": "view_cart_images", "title": "🖼️ Ver Imágenes"}
    ]
    
    full_message = f"{cart_message}\n\n¿Qué deseas hacer ahora?"
    return send_whatsapp_buttons(from_number, full_message, cart_buttons)

async def handle_payment_process(session, user_phone):
    """Procesa el pago y calcula total del carrito"""
    
    # Obtener items del carrito
    result = await session.execute(
        select(Cart).where(Cart.user_phone == user_phone)
    )
    cart_items = result.scalars().all()
    
    if not cart_items:
        return "❌ Tu carrito está vacío. Agrega productos primero."
    
    # Calcular total
    total = sum(item.product_price for item in cart_items)
    
    # Crear resumen del pedido
    items_summary = []
    for item in cart_items:
        items_summary.append(f"• {item.product_name} - PEN {item.product_price}")
    
    items_text = "\n".join(items_summary)
    
    # 🔥 NUEVO: Vaciar carrito después de procesar pago
    try:
        from sqlalchemy import delete
        # Eliminar todos los items del carrito
        await session.execute(
            delete(Cart).where(Cart.user_phone == user_phone)
        )
        await session.commit()
        print(f"🛒 Carrito vaciado automáticamente después del pago para {user_phone}")
    except Exception as e:
        print(f"❌ Error al vaciar carrito: {e}")
    
    payment_message = f"""🎉 **¡COMPRA PROCESADA CON ÉXITO!**

📋 **Resumen de tu pedido:**
{items_text}

💰 **Total a pagar: PEN {total:.2f}**

📱 **Opciones de pago:**

**💸 Yape/Plin:**
Número: +51 957670299
Nombre: HD Company

**🏪 Pago presencial:**
Visítanos en nuestra tienda física

📞 **Próximos pasos:**
1. Te contactaremos en las próximas horas
2. Coordinaremos entrega y forma de pago final
3. ¡Gracias por confiar en HD Company!

🚀 ¿Necesitas algo más o quieres seguir viendo productos?"""
    
    return payment_message

async def handle_add_to_cart_action(session, conv_id, user_phone, products):
    """Maneja toda la lógica de agregar al carrito de forma unificada"""
    # Buscar último producto seleccionado
    selected_product = await find_last_selected_product(session, conv_id, products)
    
    if not selected_product:
        return {
            "type": "text", 
            "body": "❌ No pude identificar el producto que quieres agregar. Selecciona un producto primero."
        }
    
    # Agregar al carrito
    success, message = await add_product_to_cart(session, user_phone, selected_product)
    
    if not success:
        return {"type": "text", "body": f"❌ {message}"}
    
    # Obtener resumen del carrito
    cart_summary, total = await get_cart_summary(session, user_phone)
    full_message = f"{message}\n\n{cart_summary}"
    
    # Enviar botones de confirmación
    if send_cart_confirmation_buttons(user_phone, full_message):
        # Registrar mensaje en BD
        bot_message = Message(
            conversation_id=conv_id,
            sender="agent", 
            message="[Botones post-carrito enviados]",
            timestamp=datetime.now(timezone.utc)
        )
        session.add(bot_message)
        await session.commit()
        return {"status": "interactive_sent"}
    else:
        # Fallback si fallan los botones
        return {
            "type": "text", 
            "body": f"{full_message}\n\nEscribe: 'pagar', 'seguir' o 'imágenes'"
        }

def detect_add_to_cart_intent(user_input):
    """Detecta si el usuario quiere agregar algo al carrito"""
    add_to_cart_phrases = [
        "agregar al carrito", "agregalo al carrito", "agregar carrito",
        "añadir al carrito", "lo quiero", "compralo", "agregalo",
        "quiero comprarlo", "me lo llevo", "añadelo"
    ]
    
    user_input_normalized = normalize_text(user_input)
    
    for phrase in add_to_cart_phrases:
        if phrase in user_input_normalized:
            return True
    
    return False
# =============================================================================
# FUNCIONES ADICIONALES PARA GESTIÓN DEL CARRITO
# =============================================================================

async def clear_entire_cart(session, user_phone):
    """Vacía completamente el carrito del usuario"""
    try:
        # Eliminar todos los items del carrito
        await session.execute(
            Cart.__table__.delete().where(Cart.user_phone == user_phone)
        )
        await session.commit()
        return True, "🗑️ **Carrito vaciado completamente** ✅\n\n¡Tu carrito está ahora vacío!"
    except Exception as e:
        print(f"Error vaciando carrito: {e}")
        return False, "❌ Error al vaciar el carrito. Inténtalo de nuevo."

async def remove_cart_item_by_position(session, user_phone, position):
    """Elimina un producto del carrito por su posición"""
    try:
        # Obtener todos los items del carrito
        cart_items = await session.execute(
            select(Cart).where(Cart.user_phone == user_phone).order_by(Cart.added_at)
        )
        cart_items = cart_items.scalars().all()
        
        if not cart_items:
            return False, "🛒 Tu carrito está vacío."
        
        if position < 1 or position > len(cart_items):
            return False, f"❌ Posición inválida. Tu carrito tiene {len(cart_items)} productos (1-{len(cart_items)})."
        
        # Obtener el producto a eliminar (posición - 1 para índice)
        item_to_remove = cart_items[position - 1]
        product_name = item_to_remove.product_name
        
        # Eliminar el producto
        await session.delete(item_to_remove)
        await session.commit()
        
        return True, f"🗑️ **Producto eliminado:**\n❌ {product_name}\n\n✅ Eliminado de tu carrito exitosamente."
    except Exception as e:
        print(f"Error eliminando producto del carrito: {e}")
        return False, "❌ Error al eliminar el producto. Inténtalo de nuevo."

async def get_numbered_cart_summary(session, user_phone):
    """Obtiene el resumen del carrito con posiciones numeradas para eliminación"""
    cart_items = await session.execute(
        select(Cart).where(Cart.user_phone == user_phone).order_by(Cart.added_at)
    )
    cart_items = cart_items.scalars().all()
    
    if not cart_items:
        return "🛒 Tu carrito está vacío.", 0
    
    cart_message = "🛒 **Productos en tu carrito:**\n\n"
    total_price = 0
    
    for i, item in enumerate(cart_items, 1):
        cart_message += f"**{i}.** {item.product_name}\n"
        cart_message += f"    💰 Precio: PEN {item.product_price}\n\n"
        total_price += item.product_price
    
    cart_message += f"💳 **Total: PEN {total_price:.2f}**\n\n"
    cart_message += "Escribe:\n"
    cart_message += "🗑️ 'Eliminar 1', 'Eliminar 2', etc.\n"
    cart_message += "🗑️ 'o Vaciar carrito' para eliminar todo\n"
    cart_message += "o Selecciona una opción:\n"

    return cart_message, total_price

def send_cart_management_options(from_number, cart_message):
    """Envía opciones de gestión del carrito simplificadas (SIN Ver Numerado)"""
    cart_options = [
        {"id": "pagar", "title": "💳 Pagar Ahora", "description": "Proceder con el pago del pedido"},
        {"id": "clear_cart", "title": "🗑️ Vaciar Todo", "description": "Eliminar todos los productos"},
        {"id": "continue_shopping", "title": "🛍️ Seguir", "description": "Continuar viendo productos"},
        {"id": "view_cart_images", "title": "🖼️ Ver Imágenes", "description": "Ver imágenes de productos"}
    ]
    
    full_message = f"{cart_message}\n\n¿Qué deseas hacer con tu carrito?"
    
    # Intentar enviar como lista
    if send_whatsapp_list(from_number, full_message, cart_options):
        return True
    
    # Fallback a botones
    short_buttons = [
        {"id": "pagar", "title": "💳 Pagar"},
        {"id": "clear_cart", "title": "🗑️ Vaciar"},
        {"id": "continue_shopping", "title": "🛍️ Seguir"}
    ]
    
    return send_whatsapp_buttons(from_number, full_message, short_buttons)

async def handle_cart_management(session, conv_id, user_phone, management_intent):
    """Maneja todas las acciones de gestión del carrito (VERSIÓN MEJORADA)"""
    
    if management_intent["action"] == "clear_all":
        success, message = await clear_entire_cart(session, user_phone)
        
        if success:
            continue_buttons = [
                {"id": "continue_shopping", "title": "🛍️ Seguir"},
                {"id": "view_main_menu", "title": "🏠 Menú"}
            ]
            
            if send_whatsapp_buttons(user_phone, message, continue_buttons):
                return {"status": "interactive_sent", "message": message}
            else:
                return {"type": "text", "body": f"{message}\n\nEscribe 'menu' para ver opciones."}
        else:
            return {"type": "text", "body": message}
    
    elif management_intent["action"] == "remove_position":
        position = management_intent["position"]
        success, message = await remove_cart_item_by_position(session, user_phone, position)
        
        if success:
            # Obtener carrito actualizado
            updated_cart, total = await get_numbered_cart_summary(session, user_phone)
            full_message = f"{message}\n\n{updated_cart}"
            
            # Si aún hay productos, mostrar opciones de gestión
            if total > 0:
                if send_cart_management_options(user_phone, full_message):
                    return {"status": "interactive_sent", "message": full_message}
                else:
                    return {"type": "text", "body": f"{full_message}\n\nComandos:\n🗑️ 'vaciar carrito'\n🖼️ 'imágenes'\n🛍️ 'seguir'"}
            else:
                # Si el carrito está vacío, mostrar mensaje simple
                return {"type": "text", "body": f"{message}\n\n🛒 Tu carrito está ahora vacío."}
        else:
            return {"type": "text", "body": message}
        
    elif management_intent["action"] == "view_cart":
        cart_summary, total = await get_numbered_cart_summary(session, user_phone)
        
        if total > 0:
            # Usar la nueva función de lista en lugar de botones
            if send_cart_management_options(user_phone, cart_summary):
                return {"status": "interactive_sent", "message": cart_summary}
            else:
                return {"type": "text", "body": f"{cart_summary}\n\nComandos:\n🗑️ 'vaciar carrito'\n🖼️ 'imágenes'\n🛍️ 'seguir'"}
        else:
            return {"type": "text", "body": cart_summary}
    
    return {"type": "text", "body": "❌ Acción no reconocida."}

def detect_cart_management_intent(user_input):
    """Detecta intenciones de gestión del carrito"""
    user_input_normalized = normalize_text(user_input)
    
    # Detectar comando para vaciar carrito
    clear_phrases = [
        "vaciar carrito", "limpiar carrito", "borrar carrito", 
        "eliminar todo", "vaciar todo", "clear cart"
    ]
    
    for phrase in clear_phrases:
        if phrase in user_input_normalized:
            return {"action": "clear_all"}
    
    # Detectar comando para eliminar por posición
    remove_patterns = [
        r"eliminar\s+(\d+)",
        r"borrar\s+(\d+)", 
        r"quitar\s+(\d+)",
        r"remove\s+(\d+)"
    ]
    
    for pattern in remove_patterns:
        match = re.search(pattern, user_input_normalized)
        if match:
            try:
                position = int(match.group(1))
                return {"action": "remove_position", "position": position}
            except ValueError:
                continue
    
    # Detectar comando para ver carrito
    view_phrases = [
        "ver carrito", "mostrar carrito", "carrito", "mi carrito",
        "que tengo en el carrito", "carrito numerado"
    ]
    
    for phrase in view_phrases:
        if phrase in user_input_normalized:
            return {"action": "view_cart"}
    
    return None
# =============================================================================
# FUNCIONES DE PRODUCTOS
# =============================================================================

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
        # Filtrado específico para marcas de laptops
        elif "laptop" in query_lower and any(brand in query_lower for brand in ["hp", "lenovo", "dell", "asus"]):
            brand = next(brand for brand in ["hp", "lenovo", "dell", "asus"] if brand in query_lower)
            for product in products:
                if ("laptop" in product.get("nombre", "").lower() and 
                    brand in product.get("nombre", "").lower()):
                    found_products.append(product)

        # Filtrado específico para tipos de impresoras
        elif "impresora" in query_lower and any(tipo in query_lower for tipo in ["multifuncional", "termica", "laser", "tinta"]):
            tipo = next(tipo for tipo in ["multifuncional", "termica", "laser", "tinta"] if tipo in query_lower)
            for product in products:
                if ("impresora" in product.get("nombre", "").lower() and 
                    tipo in product.get("nombre", "").lower()):
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
    response = f"📋 Productos disponibles en {query.title()}:\n\n"
    for i, product in enumerate(products_list[:5], 1):
        response += f"{i}. 💻 **{product['nombre']}**\n"
        response += f"   💰 Precio: {product['precio']}\n"
        response += f"   📝 {product['descripcion']}\n\n"
    if len(products_list) > 5:
        response += f"Y {len(products_list) - 5} productos más disponibles.\n"
    response += "¿Te interesa alguno en particular? ¿Necesitas más información?"
    return response

def extract_products_from_message(message_text):
    """Extrae productos de un mensaje que contiene una lista formateada"""
    products_in_list = []
    lines = message_text.split('\n')
    
    current_product = {}
    for line in lines:
        line = line.strip()
        
        # Detectar línea de producto (número + nombre)
        product_match = re.match(r'(\d+)\.\s*💻\s*\*\*(.+?)\*\*', line)
        if product_match:
            if current_product:  # Guardar producto anterior si existe
                products_in_list.append(current_product)
            
            current_product = {
                "position": int(product_match.group(1)),
                "name": product_match.group(2).strip(),
                "price": "",
                "description": ""
            }
        
        # Detectar precio
        elif line.startswith('💰') and current_product:
            price_match = re.search(r'(PEN\s*[\d,]+(?:\.\d{2})?)', line)
            if price_match:
                current_product["price"] = price_match.group(1)
        
        # Detectar descripción
        elif line.startswith('📝') and current_product:
            current_product["description"] = line.replace('📝', '').strip()
    
    # Agregar último producto
    if current_product:
        products_in_list.append(current_product)
    
    return products_in_list

async def get_last_product_list(session, conv_id):
    """Obtiene la última lista de productos enviada (VERSIÓN FINAL)"""
    messages = await session.execute(
        select(Message).where(Message.conversation_id == conv_id)
        .order_by(Message.timestamp.desc()).limit(20)
    )
    messages = messages.scalars().all()
    
    for msg in messages:
        if (msg.sender == 'agent' and 
            isinstance(msg.message, str) and
            ('💰 Precio:' in msg.message or 'PEN ' in msg.message) and  # ← CAMBIO AQUÍ
            ('**' in msg.message)):  # ← CAMBIO AQUÍ - MÁS GENERAL
            
            print(f"🔍 EVALUANDO MENSAJE: {msg.message[:100]}...")
            
            # PRIORIDAD 1: Listas del LLM 
            # PRIORIDAD 1: Listas del LLM 
            if ('🌟' in msg.message or 'tenemos' in msg.message):
                
                print(f"🎯 ENCONTRADA LISTA LLM: {msg.message[:100]}...")
                print(f"🔍 MENSAJE COMPLETO LLM: {msg.message}")
                
                # EXTRACCIÓN UNIVERSAL DE PRODUCTOS DEL LLM
                products_list = []
                
                # MÉTODO 1: Formato con números (1. emoji **nombre**)
                llm_products_numbered = re.findall(r'(\d+)\.\s*[💻🖨️🖱️📱]\s*\*\*(.+?)\*\*.*?💰 Precio:\s*(PEN[\d\s,\.]+)', msg.message, re.DOTALL)
                
                if llm_products_numbered:
                    print("📋 Formato numerado detectado")
                    for pos, name, price in llm_products_numbered:
                        products_list.append({
                            "position": int(pos),
                            "name": name.strip(),
                            "price": price.strip(),
                            "description": ""
                        })
                else:
                    # MÉTODO 2: Formato con guiones (- **nombre**)
                    llm_products_dash = re.findall(r'-\s*\*\*(.+?)\*\*\s*-\s*(PEN[\d\s,\.]+)', msg.message)
                    
                    if llm_products_dash:
                        print("📋 Formato con guiones detectado")
                        for i, (name, price) in enumerate(llm_products_dash, 1):
                            products_list.append({
                                "position": i,
                                "name": name.strip(),
                                "price": price.strip(),
                                "description": ""
                            })
                
                if products_list:
                    print(f"✅ PRODUCTOS LLM EXTRAÍDOS (UNIVERSAL): {len(products_list)} productos")
                    for p in products_list:
                        print(f"   {p['position']}. {p['name']} - {p['price']}")
                    
                    return {
                        "category": "LLM_UNIVERSAL",
                        "products": products_list,
                        "message": msg.message
                    }
                else:
                    print("⚠️ Lista LLM encontrada pero no se pudieron extraer productos con ningún formato")
            
            # PRIORIDAD 2: Listas hardcoded
            elif ('Productos disponibles en' in msg.message and 
                  re.search(r'\d+\.\s*[💻🖨️]\s*\*\*', msg.message)):
                
                print(f"📋 ENCONTRADA LISTA HARDCODED: {msg.message[:100]}...")
                
                # Tu código hardcoded existente...
                numbered_products = re.findall(r'(\d+)\.\s*[💻🖨️]\s*\*\*(.+?)\*\*.*?💰 Precio:\s*(PEN[\d\s,\.]+)', msg.message, re.DOTALL)
                products_list = []
                for pos, name, price in numbered_products:
                    products_list.append({
                        "position": int(pos),
                        "name": name.strip(),
                        "price": price.strip(),
                        "description": ""
                    })
                
                if products_list:
                    print(f"✅ PRODUCTOS HARDCODED EXTRAÍDOS: {len(products_list)} productos")
                    return {
                        "category": "hardcoded",
                        "products": products_list,
                        "message": msg.message
                    }
    
    print("❌ NO SE ENCONTRÓ LISTA DE PRODUCTOS")
    return None

def detect_position_selection(user_input):
    """Detecta si el usuario está seleccionando por posición"""
    input_lower = user_input.lower().strip()
    
    # Patrones de selección
    patterns = [
        (r'(?:quiero\s+)?(?:el\s+)?primero?', 1),
        (r'(?:quiero\s+)?(?:el\s+)?segundo?', 2),
        (r'(?:quiero\s+)?(?:el\s+)?tercero?', 3),
        (r'(?:quiero\s+)?(?:el\s+)?cuarto?', 4),
        (r'(?:quiero\s+)?(?:el\s+)?quinto?', 5),
        (r'(?:quiero\s+)?(?:el\s+)?(\d+)', None),  # Número directo
        (r'^(\d+)$', None),  # Solo número
    ]
    
    for pattern, position in patterns:
        match = re.search(pattern, input_lower)
        if match:
            if position is None:  # Caso de número directo
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
            else:
                return position
    
    return None

async def analyze_user_intent_with_llm(user_input, recent_messages=None):
    """Usa LLM para analizar la intención del usuario"""
    
    # Contexto de mensajes recientes
    context = ""
    if recent_messages:
        context = "\n".join([f"{msg.sender}: {msg.message}" for msg in recent_messages[-3:]])
    
    intent_prompt = f"""
Analiza esta entrada del usuario y determina su intención EXACTA:

Entrada del usuario: "{user_input}"
Contexto reciente de conversación:
{context}

Responde SOLO con UNA de estas opciones:
- SELECTION: Si está seleccionando de una lista mostrada recientemente (ej: "1", "el primero", "quiero el segundo")
- NEW_QUERY: Si está haciendo una consulta nueva (ej: "¿cuál es mejor?", "necesito para diseño", "presupuesto 500")
- PRODUCT_SEARCH: Si busca productos específicos (ej: "laptops HP", "impresoras Canon")
- COMPARISON: Si quiere comparar productos (ej: "HP vs Lenovo", "cuál es mejor")

REGLAS:
- Solo SELECTION si el contexto muestra una lista numerada reciente
- Todo lo demás que sea consulta/pregunta → NEW_QUERY, PRODUCT_SEARCH o COMPARISON
- NO confundir consultas con selecciones

Responde SOLO la categoría, nada más.
"""

    try:
        response = await llm.ainvoke([{"role": "user", "content": intent_prompt}])
        intent = response.content.strip().upper()
        
        print(f"🧠 LLM Intent Analysis: '{user_input}' → {intent}")
        
        if intent in ["SELECTION", "NEW_QUERY", "PRODUCT_SEARCH", "COMPARISON"]:
            return intent
        else:
            return "NEW_QUERY"  # Default seguro
            
    except Exception as e:
        print(f"❌ Error en análisis de intención: {e}")
        return "NEW_QUERY"  # Default seguro

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

async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(lambda: SQLAlchemyUserDatabase(User, AsyncSessionLocal))):
    yield UserManager(user_db)

fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

auth_router = fastapi_users.get_auth_router(auth_backend)
current_user = fastapi_users.current_user(active=True)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =============================================================================
# CONFIGURACIÓN DE LLM
# =============================================================================

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

user_chat_histories = {}

def get_chat_history(user_phone: str):
    """Obtiene o crea el historial de chat para un usuario específico"""
    if user_phone not in user_chat_histories:
        user_chat_histories[user_phone] = InMemoryChatMessageHistory()
    return user_chat_histories[user_phone]

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

recommend_task = Task(
    description="Basado en la entrada del usuario, recomienda 2-3 productos de la lista proporcionada. Incluye nombre, precio y descripción.",
    agent=sales_agent,
    expected_output="Una lista de 2-3 productos recomendados con sus nombres, precios y descripciones."
)

sales_crew = Crew(
    agents=[sales_agent],
    tasks=[recommend_task],
    verbose=True
)

# =============================================================================
# MODELOS PYDANTIC
# =============================================================================

class WhatsAppMessage(BaseModel):
    text: str | None = None
    from_: str | None = Field(None, alias="from")
    message_type: str = "text"
    from_number: str | None = None
    message_body: str | None = None
    button_id: str | None = None
    list_reply: dict | None = None
    interactive: dict | None = None

    def get_message_content(self) -> str | None:
        if self.interactive and "button_reply" in self.interactive:
            return self.interactive["button_reply"]["id"]
        elif self.interactive and "list_reply" in self.interactive:
            return self.interactive["list_reply"]["id"]
        elif self.message_type == "list_type" and self.list_reply:
            return self.list_reply.get("id")
        return self.message_body or self.text
    
    def get_from_number(self) -> str:
        return (self.from_number or getattr(self, "from_", None) or "").strip()

class DashboardMessage(BaseModel):
    user_phone: str
    message: str

# =============================================================================
# CONFIGURACIÓN DE FASTAPI Y SOCKET.IO
# =============================================================================

app = FastAPI()
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app.mount("/socket.io", socketio.ASGIApp(sio))

app.include_router(auth_router, prefix="/auth")

# =============================================================================
# EVENTOS DE SOCKET.IO
# =============================================================================

@sio.event
async def connect(sid, environ):
    print(f"Cliente conectado: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Cliente desconectado: {sid}")

# =============================================================================
# FUNCIONES DE BASE DE DATOS
# =============================================================================

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("Tablas creadas:", Base.metadata.tables.keys())
    async with AsyncSessionLocal() as session:
        user = await session.execute(User.__table__.select().where(User.username == "admin"))
        if not user.scalars().first():
            hashed_password = pwd_context.hash("admin123")
            session.add(User(username="admin", hashed_password=hashed_password))
            await session.commit()

# =============================================================================
# RUTAS WEB (LOGIN, DASHBOARD, ETC.)
# =============================================================================

@app.get("/")
async def root():
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

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
    
    jwt_strategy = get_jwt_strategy()
    token = await jwt_strategy.create_access_token(data={"sub": str(user.id)})
    cookie_transport.write(response, token)
    
    return RedirectResponse(url="/dashboard", status_code=303)

@app.get("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(cookie_transport.cookie_name)
    return response

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
            timestamp=datetime.now(timezone.utc)
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_poll": None
        })
        return {"status": "Mensaje enviado"}

# =============================================================================
# ENDPOINT PRINCIPAL: PROCESAR MENSAJES DE WHATSAPP
# =============================================================================

@app.post("/process")
async def process_message(message: WhatsAppMessage):
    try:
        print(f"Datos recibidos: {message.model_dump()}")
        user_input = message.get_message_content() if message.message_type in ["text", "button"] else ""
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
                    timestamp=datetime.now(timezone.utc)
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })

            # =============================================================
            # LÓGICA PRINCIPAL DE PROCESAMIENTO
            # =============================================================
            
            # 1. DETECTAR SALUDOS Y ENVIAR MENÚ PRINCIPAL
            normalized_input = normalize_text(user_input)
            if normalized_input in greetings or any(greet in normalized_input for greet in greetings):
                welcome_message = f"¡Hola{ ' de nuevo' if 'nuevo' in normalized_input else ''}, {conv.name}! 😊 Bienvenido a HD Company. ¿En qué te puedo ayudar hoy?"
                if send_whatsapp_list(from_number, welcome_message, main_list_options):
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent", 
                        message="[Lista de opciones principales enviada]",
                        timestamp=datetime.now(timezone.utc)
                    )
                    session.add(bot_message)
                    await session.commit()
                    return {"status": "interactive_sent"}
                else:
                    response_body = f"{welcome_message}\n\nEscribe una opción:\n🔥 Ofertas\n💻 Laptops\n🖨️ Impresoras\n🖱️ Accesorios\n🛠️ Soporte\n👤 Agente"
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_body,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_body,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_body}
            
            # 2. MANEJAR ESTADO INICIAL (pedir nombre)
            if client.name == "Desconocido":
                if user_input and message.message_type == "text":
                    if not user_input.lower() in ["hola", "hi", "hey", "buenas", "hello"]:
                        client.name = user_input.strip()
                        conv.name = user_input.strip()
                        session.add(client)
                        session.add(conv)
                        await session.commit()
                        
                        welcome_message = f"¡Hola, {client.name}! 😊 Bienvenido a HD Company. ¿En qué te puedo ayudar hoy?"
                        
                        if send_whatsapp_list(from_number, welcome_message, main_list_options):
                            bot_message = Message(
                                conversation_id=conv.id,
                                sender="agent", 
                                message="[Lista de opciones principales enviada]",
                                timestamp=datetime.now(timezone.utc)
                            )
                            session.add(bot_message)
                            await session.commit()
                            return {"status": "interactive_sent"}
                        else:
                            response_body = {
                                "type": "text", 
                                "body": f"{welcome_message}\n\nEscribe una opción:\n🔥 Ofertas\n💻 Laptops\n🖨️ Impresoras\n🖱️ Accesorios\n🛠️ Soporte\n👤 Agente"
                            }
                    else:
                        response_body = {"type": "text", "body": "😊 Por favor, dime tu nombre para continuar."}
                else:
                    response_body = {"type": "text", "body": "😊 ¡Hola! Soy el asistente de HD Company. ¿Cuál es tu nombre?"}

                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=response_body["body"],
                    timestamp=datetime.now(timezone.utc)
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
                    "message": response_body["body"],
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return response_body

            # 3. **DETECCIÓN MEJORADA DE AGREGAR AL CARRITO** - VERSIÓN CORREGIDA
            if detect_add_to_cart_intent(user_input):
                print(f"🛒 Detectada intención de agregar al carrito: {user_input}")
                
                # Buscar el último producto mencionado por el LLM o seleccionado
                recent_messages = await session.execute(
                    select(Message).where(Message.conversation_id == conv.id)
                    .order_by(Message.timestamp.desc()).limit(10)
                )
                recent_messages = recent_messages.scalars().all()
                
                # 🔥 NUEVO: Estrategia mejorada de búsqueda
                selected_product_name = None
                matching_product = None
                
                # Método 1: Buscar ACCION:AGREGAR_CARRITO válida (con nombre real de producto)
                for msg in recent_messages:
                    if msg.sender == "agent" and "ACCION:AGREGAR_CARRITO:" in msg.message:
                        product_candidate = msg.message.split("ACCION:AGREGAR_CARRITO:")[1].strip()
                        print(f"🔍 Candidato ACCION: {product_candidate}")
                        
                        # Verificar que sea un nombre de producto real, no una categoría
                        if not any(invalid in product_candidate.lower() for invalid in ["productos en", "laptops", "impresoras", ":"]):
                            selected_product_name = product_candidate
                            print(f"✅ ACCION válida encontrada: {selected_product_name}")
                            break
                
                # Método 2: Buscar en mensajes del LLM con productos específicos
                if not selected_product_name:
                    for msg in recent_messages:
                        if (msg.sender == "agent" and 
                            any(keyword in msg.message.lower() for keyword in ["excelente elección", "aquí tienes", "recomiendo"]) and
                            "**Laptop" in msg.message):
                            
                            # Extraer primer producto con formato **Laptop...**
                            import re
                            matches = re.findall(r'\*\*(Laptop[^*]+)\*\*', msg.message)
                            if matches:
                                selected_product_name = matches[0].strip()
                                print(f"✅ Producto de LLM encontrado: {selected_product_name}")
                                break
                
                # Método 3: Buscar último producto en lista hardcoded
                if not selected_product_name:
                    last_list = await get_last_product_list(session, conv.id)
                    if last_list and last_list.get("products"):
                        # Tomar el primero de la lista como default
                        selected_product_name = last_list["products"][0]["name"]
                        print(f"🎯 Usando primer producto de lista: {selected_product_name}")
                
                # Buscar producto completo en inventario
                if selected_product_name:
                    for product in products:
                        if match_product_name(selected_product_name, product['nombre']):
                            matching_product = product
                            print(f"✅ Producto coincidente encontrado: {product['nombre']}")
                            break
                    
                    # Si no hay coincidencia exacta, buscar por palabras clave
                    if not matching_product:
                        selected_words = selected_product_name.lower().split()
                        for product in products:
                            product_words = product['nombre'].lower().split()
                            common_words = len(set(selected_words) & set(product_words))
                            if common_words >= 2:  # Al menos 2 palabras en común
                                matching_product = product
                                print(f"✅ Coincidencia por palabras clave: {product['nombre']}")
                                break
                
                if matching_product:
                    # Agregar al carrito
                    cart_item = Cart(
                        user_phone=from_number,
                        product_name=matching_product['nombre'],
                        product_price=float(matching_product['precio'].replace("PEN ", "")),
                        added_at=datetime.now(timezone.utc)
                    )
                    session.add(cart_item)
                    await session.commit()
                    
                    cart_buttons = [
                        {"id": "view_cart", "title": "🛒 Ver Carrito"},
                        {"id": "pay_now", "title": "💳 Pagar Ahora"},
                        {"id": "continue_shopping", "title": "🛍️ Seguir"}
                    ]
                    
                    success_msg = f"✅ **{matching_product['nombre'][:50]}{'...' if len(matching_product['nombre']) > 50 else ''}** agregado al carrito!\n\n💰 Precio: {matching_product['precio']}\n\n¿Qué quieres hacer ahora?"
                    
                    if send_whatsapp_buttons(from_number, success_msg, cart_buttons):
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent",
                            message=f"Producto agregado: {matching_product['nombre']}",
                            timestamp=datetime.now(timezone.utc)
                        )
                        session.add(bot_message)
                        await session.commit()
                        return {"status": "interactive_sent"}
                    else:
                        # Fallback sin botones
                        fallback_msg = f"{success_msg}\n\nEscribe: 'ver carrito', 'pagar' o 'seguir'"
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent",
                            message=fallback_msg,
                            timestamp=datetime.now(timezone.utc)
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
                            "message": fallback_msg,
                            "sender": "agent",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "active_poll": None
                        })
                        return {"type": "text", "body": fallback_msg}
                else:
                    response_text = f"❌ No pude identificar el producto específico. Intenta seleccionar un producto de la lista primero."
                
                # Error fallback
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=response_text,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": response_text}
            # 3.5. **GESTIÓN DEL CARRITO**: DETECTAR COMANDOS DE GESTIÓN
            management_intent = detect_cart_management_intent(user_input)
            if management_intent:
                print(f"🗑️ Detectada gestión de carrito: {management_intent}")
                result = await handle_cart_management(session, conv.id, from_number, management_intent)
                
                # Si se enviaron elementos interactivos, no enviar más mensajes
                if result.get("status") == "interactive_sent":
                    # Registrar en base de datos
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent", 
                        message=f"[Gestión de carrito: {management_intent['action']}]",
                        timestamp=datetime.now(timezone.utc)
                    )
                    session.add(bot_message)
                    await session.commit()
                    return result
                
                # Enviar respuesta normal y emitir por socket
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=result["body"],
                    timestamp=datetime.now(timezone.utc)
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
                    "message": result["body"],
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return result
            # 4. MANEJAR ESTADO ESCALADO
            if conv.escalated == "True":
                if user_input.lower() == "volver":
                    conv.escalated = "False"
                    conv.state = "active"
                    await session.commit()
                    
                    back_message = f"¡Perfecto, {conv.name}! 😊 ¿En qué te ayudo ahora?"
                    if send_whatsapp_list(from_number, back_message, main_list_options):
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent", 
                            message="[Lista de opciones principales enviada - regreso]",
                            timestamp=datetime.now(timezone.utc)
                        )
                        session.add(bot_message)
                        await session.commit()
                        return {"status": "interactive_sent"}
                else:
                    response_body = "🔔 Estás conectado con un agente. Escribe 'volver' para regresar al menú."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_body,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_body,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_body}

            # 5. MANEJAR BOTONES ESPECÍFICOS
            # 5. MANEJAR BOTONES ESPECÍFICOS
            if user_input == "add_to_cart":
                print(f"🛒 Botón add_to_cart presionado")
                
                # Buscar último ACCION:AGREGAR_CARRITO
                recent_messages = await session.execute(
                    select(Message).where(Message.conversation_id == conv.id)
                    .order_by(Message.timestamp.desc()).limit(10)
                )
                recent_messages = recent_messages.scalars().all()
                
                selected_product_name = None
                matching_product = None
                
                # Buscar ACCION:AGREGAR_CARRITO válida
                for msg in recent_messages:
                    if msg.sender == "agent" and "ACCION:AGREGAR_CARRITO:" in msg.message:
                        product_candidate = msg.message.split("ACCION:AGREGAR_CARRITO:")[1].strip()
                        print(f"🎯 Producto encontrado en ACCION: {product_candidate}")
                        selected_product_name = product_candidate
                        break
                
                if selected_product_name:
                    # Buscar producto completo en inventario
                    for product in products:
                        if match_product_name(selected_product_name, product['nombre']):
                            matching_product = product
                            print(f"✅ Producto coincidente: {product['nombre']}")
                            break
                
                if matching_product:
                    # Agregar al carrito
                    cart_item = Cart(
                        user_phone=from_number,
                        product_name=matching_product['nombre'],
                        product_price=float(matching_product['precio'].replace("PEN ", "")),
                        added_at=datetime.now(timezone.utc)
                    )
                    session.add(cart_item)
                    await session.commit()
                    
                    cart_buttons = [
                        {"id": "view_cart", "title": "🛒 Ver Carrito"},
                        {"id": "pay_now", "title": "💳 Pagar Ahora"},
                        {"id": "continue_shopping", "title": "🛍️ Seguir"}
                    ]
                    
                    success_msg = f"✅ **{matching_product['nombre'][:50]}{'...' if len(matching_product['nombre']) > 50 else ''}** agregado al carrito!\n\n💰 Precio: {matching_product['precio']}\n\n¿Qué quieres hacer ahora?"
                    
                    if send_whatsapp_buttons(from_number, success_msg, cart_buttons):
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent",
                            message=f"Producto agregado: {matching_product['nombre']}",
                            timestamp=datetime.now(timezone.utc)
                        )
                        session.add(bot_message)
                        await session.commit()
                        return {"status": "interactive_sent"}
                
                # Error fallback
                response_text = "❌ No pude identificar el producto que quieres agregar. Selecciona un producto primero."
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=response_text,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": response_text}
            
            # Manejar botón de vaciar carrito
            if user_input == "clear_cart":
                success, message = await clear_entire_cart(session, from_number)
                
                if success:
                    continue_buttons = [
                        {"id": "continue_shopping", "title": "🛍️ Seguir Comprando"},
                        {"id": "view_main_menu", "title": "🏠 Menú Principal"}
                    ]
                    
                    if send_whatsapp_buttons(from_number, message, continue_buttons):
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent", 
                            message="[Carrito vaciado - botones enviados]",
                            timestamp=datetime.now(timezone.utc)
                        )
                        session.add(bot_message)
                        await session.commit()
                        return {"status": "interactive_sent"}
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=message,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": message,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": message}

            # Manejar botón de menú principal
            if user_input == "view_main_menu":
                menu_message = "🏠 **Menú Principal** - ¿En qué te puedo ayudar?"
                
                if send_whatsapp_list(from_number, menu_message, main_list_options):
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent", 
                        message="[Lista de opciones principales enviada]",
                        timestamp=datetime.now(timezone.utc)
                    )
                    session.add(bot_message)
                    await session.commit()
                    return {"status": "interactive_sent"}
            # Manejar botón de ver imágenes del carrito
            if user_input == "view_cart_images":
                # Obtener productos del carrito
                cart_items = await session.execute(
                    select(Cart).where(Cart.user_phone == from_number)
                )
                cart_items = cart_items.scalars().all()
                
                if not cart_items:
                    response_text = "🛒 Tu carrito está vacío. No hay imágenes para mostrar."
                    # Return normal para texto
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
                    )
                    session.add(bot_message)
                    await session.commit()
                    return {"type": "text", "body": response_text}
                else:
                    # Buscar primer producto con imagen
                    for item in cart_items:
                        matching_product = None
                        for product in products:
                            if match_product_name(item.product_name, product['nombre']):
                                matching_product = product
                                break
                        
                        if matching_product and matching_product.get('image_url'):
                            image_link = matching_product['image_url']
                            product_name = item.product_name[:60]
                            caption = f"🖼️Imagen de {product_name}\n\n¿Te gustaría continuar con la compra😍?"
                            
                            # RETORNO ESPECIAL PARA MAKE.COM
                            bot_message = Message(
                                conversation_id=conv.id,
                                sender="agent",
                                message=f"[Imagen enviada: {product_name}]",
                                timestamp=datetime.now(timezone.utc)
                            )
                            session.add(bot_message)
                            await session.commit()
                            
                            # RETURN CON TYPE: IMAGE
                            return {
                                "type": "image",
                                "image_link": image_link,
                                "caption": caption
                            }
                    
                    # Si no hay imágenes
                    response_text = "❌ No hay imágenes disponibles para los productos en tu carrito."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
                    )
                    session.add(bot_message)
                    await session.commit()
                    return {"type": "text", "body": response_text}
            # Manejar botón de ver carrito numerado
            if user_input == "view_numbered_cart":
                cart_summary, total = await get_numbered_cart_summary(session, from_number)
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=cart_summary,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": cart_summary,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": cart_summary}
            
            # 6. MANEJAR SELECCIONES DE CATEGORÍAS Y PRODUCTOS

            # Manejar selección de laptops (SOLO para comandos exactos)
            if user_input.lower() in ["laptops", "💻 laptops"]:
                found_products = search_products("laptops", "laptops")
                if found_products:
                    response_text = format_products_response(found_products, "Laptops")
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
                else:
                    response_text = "Lo siento, no tenemos laptops disponibles en este momento."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
                
            # Manejar selección de impresoras (SOLO para comandos exactos)
            if user_input.lower() in ["impresoras", "🖨️ impresoras"]:
                found_products = search_products("impresoras", "impresoras")
                if found_products:
                    response_text = format_products_response(found_products, "Impresoras")
                    
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
                else:
                    response_text = "Lo siento, no tenemos impresoras disponibles en este momento."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
            
            # Manejar selección de ofertas
            if user_input.lower() in ["ofertas", "🔥 ofertas especiales", "ofertas especiales"]:
                found_products = search_products("ofertas")
                if found_products:
                    response_text = format_products_response(found_products, "Ofertas Especiales")
                    
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
                else:
                    response_text = "No hay ofertas especiales disponibles en este momento."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
            
            # Manejar selección de accesorios
            if user_input.lower() in ["accesorios", "🖱️ accesorios"]:
                found_products = search_products("accesorios", "accesorios")
                if found_products:
                    response_text = format_products_response(found_products, "Accesorios")
                    
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
                else:
                    response_text = "No tenemos accesorios disponibles en este momento."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}

            # Manejar soporte técnico
            if user_input.lower() in ["soporte", "🛠️ soporte técnico", "soporte tecnico"]:
                response_text = """🛠️ **Soporte Técnico** - ¿En qué puedo ayudarte?

            Estoy aquí para ayudarte con:
            • 🔧 Problemas técnicos
            • 📦 Instalación de productos  
            • 🛡️ Garantías y devoluciones
            • 💬 Consultas especializadas

            📅 **¿Necesitas atención personalizada?**
            👉 Agenda una cita: https://calendly.com/josemarianoludenalimas/agendar-cita

            ¿En qué te puedo ayudar específicamente?"""
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=response_text,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": response_text}

            # Manejar escalación a agente
            if user_input.lower() in ["agente", "👤 hablar con agente", "hablar con agente"]:
                conv.escalated = "True"
                conv.state = "escalated"
                session.add(conv)
                await session.commit()
                
                response_text = f"🔔 ¡Perfecto, {conv.name}! Te he conectado con uno de nuestros agentes humanos. En breve te atenderán.\n\nMientras tanto, puedes escribir tu consulta y la verán cuando estén disponibles.\n\nEscribe 'volver' si quieres regresar al menú automático."
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=response_text,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": response_text}

            # ================================================================
            # ANÁLISIS INTELIGENTE DE INTENCIÓN CON LLM - NUEVO
            # ================================================================
            recent_messages = await session.execute(
                select(Message).where(Message.conversation_id == conv.id)
                .order_by(Message.timestamp.desc()).limit(5)
            )
            recent_messages = recent_messages.scalars().all()

            user_intent = await analyze_user_intent_with_llm(user_input, recent_messages)
            print(f"🎯 Intención detectada: {user_intent}")

            # SI LA INTENCIÓN NO ES SELECCIÓN, IR DIRECTAMENTE AL LLM
            if user_intent in ["NEW_QUERY", "PRODUCT_SEARCH", "COMPARISON"]:
                print(f"🧠 Saltando a LLM - Intención: {user_intent}")
                # Saltar todo el bloque de selección y botones, ir directo al LLM
                pass  # Continuará al try: del LLM al final
            else:
                # Solo ejecutar lógica de selección si la intención ES selección
                print(f"🔢 Procesando como selección - Intención: {user_intent}")

            # Detectar selección por posición SOLO si la intención es SELECTION
            if user_intent == "SELECTION":
                position = detect_position_selection(user_input)
                if position is not None:
                    last_list = await get_last_product_list(session, conv.id)
                    if last_list and 1 <= position <= len(last_list["products"]):  # ← ARREGLADO
                        selected = last_list["products"][position - 1]
                        product_name = selected["name"]
                        
                        # Buscar producto completo
                        matching_product = None
                        for product in products:
                            if match_product_name(product_name, product['nombre']):
                                matching_product = product
                                break
                        
                        if matching_product:
                            # Enviar botones del producto
                            product_buttons = [
                                {"id": "add_to_cart", "title": "🛒 Agregar al Carrito"},
                                {"id": "more_info", "title": "ℹ️ Más Información"},
                                {"id": "back_to_list", "title": "⬅️ Volver a Lista"}
                            ]
                            
                            selection_msg = f"Has seleccionado **{product_name}**\n💰 Precio: {selected['price']}\n\n¿Qué te gustaría hacer?"
                            
                            if send_whatsapp_buttons(from_number, selection_msg, product_buttons):
                                bot_message = Message(
                                    conversation_id=conv.id,
                                    sender="agent",
                                    message=f"Botones de producto enviados para **{product_name}**",
                                    timestamp=datetime.now(timezone.utc)
                                )
                                session.add(bot_message)
                                await session.commit()
                                return {"status": "interactive_sent"}
                            else:
                                # Fallback
                                fallback_msg = f"{selection_msg}\n\nEscribe:\n🛒 'agregar' para agregar al carrito\nℹ️ 'info' para más información\n⬅️ 'lista' para volver"
                                
                                bot_message = Message(
                                    conversation_id=conv.id,
                                    sender="agent",
                                    message=fallback_msg,
                                    timestamp=datetime.now(timezone.utc)
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
                                    "message": fallback_msg,
                                    "sender": "agent",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "active_poll": None
                                })
                                return {"type": "text", "body": fallback_msg}
                        else:
                            response_text = f"❌ No pude encontrar el producto '{product_name}' en nuestro inventario."
                            bot_message = Message(
                                conversation_id=conv.id,
                                sender="agent",
                                message=response_text,
                                timestamp=datetime.now(timezone.utc)
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
                                "message": response_text,
                                "sender": "agent",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "active_poll": None
                            })
                            return {"type": "text", "body": response_text}
                    else:
                        response_text = "❌ Selección inválida. Por favor, elige un número de la lista mostrada."
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent",
                            message=response_text,
                            timestamp=datetime.now(timezone.utc)
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
                            "message": response_text,
                            "sender": "agent",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "active_poll": None
                        })
                        return {"type": "text", "body": response_text}

            # ================================================================
            # BOTONES Y ACCIONES INDEPENDIENTES DE SELECCIÓN ← MOVER AQUÍ
            # ================================================================

            # Manejar botón de pagar (TODAS las variaciones)
            pagar_variations = [
                "pagar", "💳 pagar ahora", "pagar ahora", 
                "💳pagar ahora", "💳pagar", "pagar ahora 💳"
            ]
            if any(variation in user_input.lower() for variation in pagar_variations):
                payment_msg = await handle_payment_process(session, from_number)
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=payment_msg,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": payment_msg,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": payment_msg}
            # 🔥 NUEVO: Manejar "ver imagen" por texto también
            image_variations = [
                "ver imagen", "ver imagenes", "🖼️ ver imagen", "imagen", "imagenes",
                "mostrar imagen", "foto", "ver foto", "picture", "ver la imagen",
                "mostrar foto", "quiero ver imagen", "quiero la imagen"
            ]

            if any(variation in user_input.lower() for variation in image_variations):
                print(f"🖼️ Detectada solicitud de imagen: {user_input}")
                
                # Buscar último producto seleccionado o mencionado
                recent_messages = await session.execute(
                    select(Message).where(Message.conversation_id == conv.id)
                    .order_by(Message.timestamp.desc()).limit(10)
                )
                recent_messages = recent_messages.scalars().all()
                
                # Buscar producto en mensajes recientes
                selected_product = None
                for msg in recent_messages:
                    if msg.sender == "agent" and "ACCION:AGREGAR_CARRITO:" in msg.message:
                        product_name = msg.message.split("ACCION:AGREGAR_CARRITO:")[1].strip()
                        selected_product = product_name
                        break
                    elif msg.sender == "agent" and "botones de producto enviados para" in msg.message.lower():
                        # Extraer de mensaje de botones
                        import re
                        match = re.search(r'\*\*(.*?)\*\*', msg.message)
                        if match:
                            selected_product = match.group(1).strip()
                            break
                
                if selected_product:
                    # Buscar imagen del producto
                    matching_product = None
                    for product in products:
                        if match_product_name(selected_product, product['nombre']):
                            matching_product = product
                            break
                    
                    if matching_product and matching_product.get('image_url'):
                        image_url = matching_product['image_url']
                        product_name = selected_product[:60]
                        caption = f"🖼️ {product_name}\n\n¿Te gustaría continuar con la compra? 😍"
                        
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent",
                            message=f"Imagen enviada: {selected_product}",
                            timestamp=datetime.now(timezone.utc)
                        )
                        session.add(bot_message)
                        await session.commit()
                        
                        # RETURN ESPECIAL PARA IMAGEN
                        return {
                            "type": "image",
                            "image_link": image_url,
                            "caption": caption
                        }
                    else:
                        response_text = f"❌ No tengo imagen disponible para **{selected_product}**."
                else:
                    response_text = "❌ Primero selecciona un producto para ver su imagen."
                
                # Respuesta de error
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=response_text,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": response_text}

            # Manejar botón "🛍️ Seguir" con emoji
            if user_input in ["continue_shopping", "🛍️ seguir", "seguir"]:
                continue_msg = "🛍️ ¡Perfecto! ¿Qué más te gustaría ver?"
                
                if send_whatsapp_list(from_number, continue_msg, main_list_options):
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent", 
                        message="[Lista de opciones principales enviada - continuar comprando]",
                        timestamp=datetime.now(timezone.utc)
                    )
                    session.add(bot_message)
                    await session.commit()
                    return {"status": "interactive_sent"}
                else:
                    fallback_msg = f"{continue_msg}\n\nEscribe una opción:\n🔥 Ofertas\n💻 Laptops\n🖨️ Impresoras\n🖱️ Accesorios\n🛠️ Soporte\n👤 Agente"
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=fallback_msg,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": fallback_msg,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": fallback_msg}

            if user_input == "continue_shopping":
                continue_msg = "🛍️ ¡Perfecto! ¿Qué más te gustaría ver?"
                
                if send_whatsapp_list(from_number, continue_msg, main_list_options):
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent", 
                        message="[Lista de opciones principales enviada - continuar comprando]",
                        timestamp=datetime.now(timezone.utc)
                    )
                    session.add(bot_message)
                    await session.commit()
                    return {"status": "interactive_sent"}
                else:
                    fallback_msg = f"{continue_msg}\n\nEscribe una opción:\n🔥 Ofertas\n💻 Laptops\n🖨️ Impresoras\n🖱️ Accesorios\n🛠️ Soporte\n👤 Agente"
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=fallback_msg,
                        timestamp=datetime.now(timezone.utc)
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
                        "message": fallback_msg,
                        "sender": "agent",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": fallback_msg}
            elif "view_cart" in user_input.lower():
                cart_summary, total = await get_numbered_cart_summary(session, from_number)
                
                if total > 0:
                    if send_cart_management_options(from_number, cart_summary):
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent", 
                            message="[Carrito mostrado con botones]",
                            timestamp=datetime.now(timezone.utc)
                        )
                        session.add(bot_message)
                        await session.commit()
                        return {"status": "interactive_sent"}
                
                # Continuar con LLM si no hay productos
            # Si llegamos aquí y NO era selección, continuar al LLM
            if user_intent in ["NEW_QUERY", "PRODUCT_SEARCH", "COMPARISON"]:
                print("🧠 Continuando al LLM para procesar consulta...")
                
            # 7. CONSULTAS GENÉRICAS CON LLM
            try:
                # Tu lógica de LLM existente...
                previous_messages = await session.execute(
                    select(Message).where(Message.conversation_id == conv.id).order_by(Message.timestamp.desc()).limit(10)
                )
                previous_messages = previous_messages.scalars().all()

                conversation_context = "Historial de conversación reciente:\n"
                for msg in reversed(previous_messages[1:]):
                    conversation_context += f"{msg.sender}: {msg.message}\n"

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
                
                cart_summary, _ = await get_cart_summary(session, from_number)

                # Detectar si es consulta de "mejor laptop" para respuesta corta
                is_best_laptop_query = any(phrase in user_input.lower() for phrase in [
                    "mejor laptop", "tu mejor laptop", "dame tu mejor", "recomienda laptop",
                    "mejor opcion", "cual es mejor", "dame una recomendacion"
                ])

                if is_best_laptop_query:
                    # Respuesta directa y corta
                    messages = [
                        {"role": "system", "content": "Eres un asistente de ventas. Responde SOLO con 1 recomendación específica."},
                        {"role": "user", "content": (
                            f"Productos disponibles: {products_context}\n\n"
                            "RESPONDE EXACTAMENTE ASÍ:\n"
                            "🏆 **MI RECOMENDACIÓN**: Te recomiendo la **[NOMBRE COMPLETO DEL PRODUCTO]** porque [razón técnica específica].\n\n"
                            "¿En qué te ayudo ahora?\n\n"
                            "IMPORTANTE:\n"
                            "- SOLO 1 producto específico\n"
                            "- NO listes múltiples opciones\n"
                            "- Máximo 50 palabras\n"
                            f"- Responde a: {user_input}"
                        )}
                    ]
                else:
                    # Respuesta normal para otras consultas
                    messages = [
                        {"role": "system", "content": create_system_prompt()},
                        {"role": "user", "content": (
                            f"Contexto de productos: {products_context}\n\n"
                            f"Contexto del carrito: {cart_summary}\n\n"
                            f"Historial de conversación: {conversation_context}\n\n"
                            f"Consulta actual del cliente: {user_input}\n\n"
                            "Responde de manera útil y amigable."
                        )}
                    ]
                
                response = await llm.ainvoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)

                # DETECTAR Y REGISTRAR LISTAS GENERADAS POR EL LLM
                if ('Productos en' in response_text and '💻 **' in response_text and '💰 Precio:' in response_text):
                    # El LLM generó una lista de productos, registrarla para futuras referencias
                    bot_message_list = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,  # Registrar la respuesta completa del LLM
                        timestamp=datetime.now(timezone.utc)
                    )
                    session.add(bot_message_list)
                    await session.commit()
                    
                    print(f"📋 Lista LLM registrada en BD para contexto futuro")

                # DETECTAR ACCIONES ESPECIALES DEL LLM
                # DETECTAR SELECCIÓN DE PRODUCTO POR LLM
                if "ACCION:SELECCIONAR_PRODUCTO:" in response_text:
                    # Extraer información
                    match = re.search(r'ACCION:SELECCIONAR_PRODUCTO:(\d+):(.+)', response_text)
                    if match:
                        position = int(match.group(1))
                        product_name = match.group(2).strip()
                        
                        # Buscar producto completo
                        matching_product = None
                        for product in products:
                            if normalize_text(product_name) in normalize_text(product['nombre']) or normalize_text(product['nombre']) in normalize_text(product_name):
                                matching_product = product
                                break
                        
                        if matching_product:
                            # Enviar botones del producto
                            product_buttons = [
                                {"id": "add_to_cart", "title": "🛒 Agregar al Carrito"},
                                {"id": "more_info", "title": "ℹ️ Más Información"},
                                {"id": "back_to_list", "title": "⬅️ Volver a Lista"}
                            ]
                            
                            selection_msg = f"Has seleccionado **{matching_product['nombre']}**\n💰 Precio: {matching_product['precio']}\n\n¿Qué te gustaría hacer?"
                            
                            if send_whatsapp_buttons(from_number, selection_msg, product_buttons):
                                bot_message = Message(
                                    conversation_id=conv.id,
                                    sender="agent",
                                    message=f"Botones de producto enviados para **{matching_product['nombre']}**",
                                    timestamp=datetime.now(timezone.utc)
                                )
                                session.add(bot_message)
                                await session.commit()
                                return {"status": "interactive_sent"}
                        
                        # Si no encuentra el producto, usar respuesta del LLM
                        response_text = response_text.replace(f"ACCION:SELECCIONAR_PRODUCTO:{position}:{product_name}", "")

                # 🔥 NUEVO: Agregar botones cuando LLM da información de producto específico
                elif ("mi recomendación" in response_text.lower() or 
                    "te recomiendo" in response_text.lower() or
                    "recomiendo la" in response_text.lower()):
                    
                    # Extraer PRIMER producto específico de la respuesta LLM
                    import re
                    
                    # Buscar tanto ** como * (asterisco simple o doble)
                    # MÉTODO 1: Buscar **MI RECOMENDACIÓN**: Te recomiendo la **PRODUCTO**
                    recom_match = re.search(r'🏆.*?mi recomendación.*?te recomiendo la \*\*([^*]+)\*\*', response_text, re.IGNORECASE | re.DOTALL)
                    if recom_match:
                        matches = [recom_match.group(1)]
                        print(f"✅ MÉTODO 1 - Recomendación encontrada: {matches[0]}")
                    else:
                        # MÉTODO 2: Buscar después de "Te recomiendo la **"
                        recom_match2 = re.search(r'te recomiendo la \*\*([^*]+)\*\*', response_text, re.IGNORECASE)
                        if recom_match2:
                            matches = [recom_match2.group(1)]
                            print(f"✅ MÉTODO 2 - Producto encontrado: {matches[0]}")
                        else:
                            # MÉTODO 3: ÚLTIMO RECURSO - Buscar primer **Laptop...**
                            matches = re.findall(r'\*\*([Ll]aptop[^*]+)\*\*', response_text)
                            if matches:
                                print(f"⚠️ MÉTODO 3 - Fallback usado: {matches[0]}")
                    
                    product_mentioned = None
                    if matches:
                        product_mentioned = matches[0].strip()
                        print(f"🎯 Producto específico extraído: {product_mentioned}")
                        
                        # Buscar producto real en inventario
                        matching_product = None
                        for product in products:
                            if match_product_name(product_mentioned, product['nombre']):
                                matching_product = product
                                break
                        
                        if matching_product:
                            # Botones para producto recomendado
                            product_buttons = [
                                {"id": "add_to_cart", "title": "🛒 Agregar"},
                                {"id": "more_info", "title": "ℹ️ Info"},
                                {"id": "continue_shopping", "title": "🛍️ Seguir"}
                            ]
                            
                            # Mensaje más corto para evitar error 1024 caracteres
                            short_response = response_text[:800] + "..." if len(response_text) > 800 else response_text
                            enhanced_msg = f"{short_response}\n\n¿Qué quieres hacer?"
                            
                            if send_whatsapp_buttons(from_number, enhanced_msg, product_buttons):
                                # Guardar NOMBRE REAL del producto para uso posterior
                                bot_message = Message(
                                    conversation_id=conv.id,
                                    sender="agent",
                                    message=f"ACCION:AGREGAR_CARRITO:{matching_product['nombre']}",
                                    timestamp=datetime.now(timezone.utc)
                                )
                                session.add(bot_message)
                                await session.commit()
                                return {"status": "interactive_sent"}

                elif "ACCION:VER_CARRITO" in response_text:
                    cart_summary, total = await get_numbered_cart_summary(session, from_number)
                    if send_cart_management_options(from_number, cart_summary):
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent", 
                            message="[Carrito mostrado por LLM]",
                            timestamp=datetime.now(timezone.utc)
                        )
                        session.add(bot_message)
                        await session.commit()
                        return {"status": "interactive_sent"}
                    else:
                        response_text = cart_summary

                elif "ACCION:MENU_PRINCIPAL" in response_text:
                    menu_message = "🏠 **Menú Principal** - ¿En qué te puedo ayudar?"
                    if send_whatsapp_list(from_number, menu_message, main_list_options):
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent", 
                            message="[Menú principal mostrado por LLM]",
                            timestamp=datetime.now(timezone.utc)
                        )
                        session.add(bot_message)
                        await session.commit()
                        return {"status": "interactive_sent"}
                    else:
                        response_text = f"{menu_message}\n\nEscribe: Ofertas, Laptops, Impresoras, Accesorios, Soporte, Agente"

                elif "ACCION:PAGAR" in response_text:
                    payment_msg = await handle_payment_process(session, from_number)
                    response_text = payment_msg


                chat_history.add_message(AIMessage(content=response_text))
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=response_text,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                
                return {"type": "text", "body": response_text}

            except Exception as e:
                print(f"Error en LLM: {e}")
                response_text = f"🛠️ Disculpa, tuve un problema procesando tu consulta. ¿Puedes intentar de nuevo? ¿En qué te ayudo, {conv.name if conv.name != 'Desconocido' else 'Ko'}?"
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=response_text,
                    timestamp=datetime.now(timezone.utc)
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_poll": None
                })
                
                return {"type": "text", "body": response_text}

    except Exception as e:
        print(f"Error en process_message: {e}")
        import traceback
        traceback.print_exc()
        return {
            "type": "text",
            "body": "Lo siento, hubo un error procesando tu mensaje. Por favor intenta de nuevo."
        }

# =============================================================================
# INICIALIZACIÓN
# =============================================================================

@app.on_event("startup")
async def run_init_db():
    print("🔧 Ejecutando init_db() desde @app.on_event startup")
    await init_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
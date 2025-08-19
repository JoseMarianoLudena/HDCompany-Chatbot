from fastapi import FastAPI, Request, Response, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
import socketio
from sqlalchemy import select
from datetime import datetime
from langchain_core.messages import AIMessage
import json



# Modulos propios
from db import AsyncSessionLocal, Conversation, Message, Client, init_db,User,Cart
from auth import auth_router, current_user, UserManager, get_user_manager, get_jwt_strategy, cookie_transport
from whatsapp import send_whatsapp_list, send_whatsapp_buttons
from cart import (
    handle_cart_management, detect_cart_management_intent,
    detect_add_to_cart_intent, clear_entire_cart, add_product_to_cart,send_cart_management_options,
    get_cart_summary, get_numbered_cart_summary, find_last_selected_product,handle_payment_process
)
from products import (
    products, faqs, search_products, format_products_response,
    get_last_product_list, detect_position_selection, analyze_user_intent_with_llm
)
from config import greetings, main_list_options
from models import WhatsAppMessage, DashboardMessage
from utils import normalize_text, match_product_name
from llm import create_system_prompt, llm, conversation, sales_agent, get_chat_history,chat_history



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
    token = await jwt_strategy.write_token(user)
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key=cookie_transport.cookie_name,
        value=token,
        max_age=cookie_transport.cookie_max_age,
        path=cookie_transport.cookie_path,
        domain=cookie_transport.cookie_domain,
        secure=cookie_transport.cookie_secure,
        httponly=cookie_transport.cookie_httponly,
        samesite=cookie_transport.cookie_samesite,
    )
    return response
    
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
                        timestamp=datetime.utcnow()
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
                        "message": response_body,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
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
                                timestamp=datetime.utcnow()
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
                    "message": response_body["body"],
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
                        added_at=datetime.utcnow()
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
                            timestamp=datetime.utcnow()
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
                            "message": fallback_msg,
                            "sender": "agent",
                            "timestamp": datetime.utcnow().isoformat(),
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
                        timestamp=datetime.utcnow()
                    )
                    session.add(bot_message)
                    await session.commit()
                    return result
                
                # Enviar respuesta normal y emitir por socket
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=result["body"],
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
                    "message": result["body"],
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
                            timestamp=datetime.utcnow()
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
                        "message": response_body,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
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
                        added_at=datetime.utcnow()
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
                            timestamp=datetime.utcnow()
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
                            timestamp=datetime.utcnow()
                        )
                        session.add(bot_message)
                        await session.commit()
                        return {"status": "interactive_sent"}
                
                bot_message = Message(
                    conversation_id=conv.id,
                    sender="agent",
                    message=message,
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
                    "message": message,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
                        timestamp=datetime.utcnow()
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
                        timestamp=datetime.utcnow()
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
                                timestamp=datetime.utcnow()
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
                        timestamp=datetime.utcnow()
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
                    "message": cart_summary,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
                else:
                    response_text = "Lo siento, no tenemos laptops disponibles en este momento."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
                else:
                    response_text = "Lo siento, no tenemos impresoras disponibles en este momento."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
                else:
                    response_text = "No hay ofertas especiales disponibles en este momento."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}
                else:
                    response_text = "No tenemos accesorios disponibles en este momento."
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": response_text}

            # ================================================================
            # ANÁLISIS INTELIGENTE DE INTENCIÓN CON LLM
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
                                    timestamp=datetime.utcnow()
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
                                    "message": fallback_msg,
                                    "sender": "agent",
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "active_poll": None
                                })
                                return {"type": "text", "body": fallback_msg}
                        else:
                            response_text = f"❌ No pude encontrar el producto '{product_name}' en nuestro inventario."
                            bot_message = Message(
                                conversation_id=conv.id,
                                sender="agent",
                                message=response_text,
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
                                "message": response_text,
                                "sender": "agent",
                                "timestamp": datetime.utcnow().isoformat(),
                                "active_poll": None
                            })
                            return {"type": "text", "body": response_text}
                    else:
                        response_text = "❌ Selección inválida. Por favor, elige un número de la lista mostrada."
                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent",
                            message=response_text,
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
                            "message": response_text,
                            "sender": "agent",
                            "timestamp": datetime.utcnow().isoformat(),
                            "active_poll": None
                        })
                        return {"type": "text", "body": response_text}

            # ================================================================
            # BOTONES Y ACCIONES INDEPENDIENTES DE SELECCIÓN
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
                    "message": payment_msg,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_poll": None
                })
                return {"type": "text", "body": payment_msg}
            # 🔥 NUEVO: Manejar "ver imagen" por texto también
            image_variations = [
                "ver imagen", "ver imagenes", "🖼️ ver imagen", "imagen", "imagenes",
                "mostrar imagen", "foto", "ver foto", "picture", "ver la imagen",
                "mostrar foto", "quiero ver imagen", "quiero la imagen"
            ]

            if any(variation in normalize_text(user_input) for variation in image_variations):
                print(f"🖼️ Detectada solicitud de imagen: {user_input}")

                # Buscar último producto seleccionado o mencionado
                recent_messages = await session.execute(
                    select(Message).where(Message.conversation_id == conv.id)
                    .order_by(Message.timestamp.desc()).limit(10)
                )
                recent_messages = recent_messages.scalars().all()

                selected_product = None
                for msg in recent_messages:
                    if msg.sender == "agent" and "ACCION:AGREGAR_CARRITO:" in msg.message:
                        product_name = msg.message.split("ACCION:AGREGAR_CARRITO:")[1].strip()
                        selected_product = product_name
                        break
                    elif msg.sender == "agent" and "botones de producto enviados para" in msg.message.lower():
                        import re
                        match = re.search(r'\*\*(.*?)\*\*', msg.message)
                        if match:
                            selected_product = match.group(1).strip()
                            break

                # Fallback: Si no hay producto seleccionado, busca el último en el carrito
                if not selected_product:
                    cart_items = await session.execute(
                        select(Cart).where(Cart.user_phone == from_number)
                        .order_by(Cart.added_at.desc())
                    )
                    cart_items = cart_items.scalars().all()
                    if cart_items:
                        selected_product = cart_items[0].product_name

                if selected_product:
                    # Buscar imagen del producto
                    matching_product = None
                    for product in products:
                        if match_product_name(selected_product, product['nombre']):
                            matching_product = product
                            break

                    if matching_product and matching_product.get('image_url'):
                        image_url = matching_product['image_url']
                        product_name = matching_product['nombre']
                        caption = f"Imagen enviada: {product_name}\n{matching_product.get('descripcion','')}\n{image_url}"

                        print(f"🖼️ RESPUESTA DE IMAGEN: {{'type': 'image', 'image_link': {image_url}, 'caption': {caption}}}")

                        bot_message = Message(
                            conversation_id=conv.id,
                            sender="agent",
                            message=f"Imagen enviada: {product_name}",
                            timestamp=datetime.utcnow()
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
                        timestamp=datetime.utcnow()
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
                        "message": fallback_msg,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
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
                        timestamp=datetime.utcnow()
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
                        "message": fallback_msg,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
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
                            timestamp=datetime.utcnow()
                        )
                        session.add(bot_message)
                        await session.commit()
                        return {"status": "interactive_sent"}
                
                # Continuar con LLM si no hay productos
            # Si llegamos aquí y NO era selección, continuar al LLM
            if user_intent in ["NEW_QUERY", "PRODUCT_SEARCH", "COMPARISON"]:
                

                ambiguous_phrases = [
                    "algo bueno", "barato", "lo mejor que tengas", "algo económico",
                    "no sé qué elegir", "ayuda a elegir", "cualquier cosa", "bueno y barato", "algo barato"
                ]
                if any(phrase in normalize_text(user_input) for phrase in ambiguous_phrases):
                    response_text = (
                        "¡Claro! ¿Podrías especificar qué tipo de producto buscas? "
                        "¿Laptop, impresora, accesorio...? Así podré ayudarte mejor."
                    )
                    bot_message = Message(
                        conversation_id=conv.id,
                        sender="agent",
                        message=response_text,
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
                        "message": response_text,
                        "sender": "agent",
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_poll": None
                    })
                    return {"type": "text", "body": response_text}    
            # 7. CONSULTAS GENÉRICAS CON LLM
            print("🧠 Continuando al LLM para procesar consulta...")
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
                        timestamp=datetime.utcnow()
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
                                    timestamp=datetime.utcnow()
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
                                    timestamp=datetime.utcnow()
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
                            timestamp=datetime.utcnow()
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
                            timestamp=datetime.utcnow()
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
                    "message": response_text,
                    "sender": "agent",
                    "timestamp": datetime.utcnow().isoformat(),
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
    from auth import pwd_context
    await init_db(pwd_context)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
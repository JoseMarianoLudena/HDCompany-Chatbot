from db import Cart, Message
from products import match_product_name, normalize_text, products
from whatsapp import send_whatsapp_buttons, send_whatsapp_list
from sqlalchemy import select, delete
from datetime import datetime
import re

# Aquí van todas las funciones de carrito, gestión y utilidades relacionadas.

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
            timestamp=datetime.utcnow()
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
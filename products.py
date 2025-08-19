import json
import re
from db import Message
from utils import normalize_text, match_product_name
#from cart import get_cart_summary  # si usas en analyze_user_intent_with_llm
#from llm import llm  # si usas en analyze_user_intent_with_llm
from sqlalchemy import select

# =============================================================================
# FUNCIONES DE PRODUCTOS
# =============================================================================

# Cargar productos y faqs desde JSON
try:
    with open("data/products.json", "r", encoding="utf-8") as f:
        products = json.load(f)
except FileNotFoundError:
    print("Error: products.json no encontrado. Creando una lista vacía.")
    products = []
except json.JSONDecodeError as e:
    print(f"Error al decodificar JSON: {e}. Creando una lista vacía.")
    products = []

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
        from llm import llm
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

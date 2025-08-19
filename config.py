# =============================================================================
# CONSTANTES Y CONFIGURACIÓN INICIAL
# =============================================================================
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

greetings = [
    "hola", "hi", "hey", "buenas", "hello", "hola de nuevo", "hola denuevo",
    "hola denuevo bro", "buen dia", "buenas tardes", "buenas noches"
]

main_list_options = [
    {"id": "ofertas", "title": "🔥 Ofertas Especiales"},
    {"id": "laptops", "title": "💻 Laptops"},
    {"id": "impresoras", "title": "🖨️ Impresoras"},
    {"id": "accesorios", "title": "🖱️ Accesorios"},
    {"id": "soporte", "title": "🛠️ Soporte Técnico"},
    {"id": "agente", "title": "👤 Hablar con Agente"}
]

# Puedes agregar aquí otras constantes globales y configuración
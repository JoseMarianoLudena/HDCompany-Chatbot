# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

import re
import unicodedata

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
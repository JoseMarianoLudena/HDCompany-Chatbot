import os
import requests
import json
# =============================================================================
# FUNCIONES DE WHATSAPP
# =============================================================================

def send_whatsapp_list(to_phone: str, message: str, list_options: list):
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
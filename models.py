from pydantic import BaseModel, Field
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

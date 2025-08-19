import os
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from crewai import Agent, Task, Crew
#from products import products, faqs  # para el prompt
# =============================================================================
# CONFIGURACIÓN DE LLM
# =============================================================================

def create_system_prompt():
    """Crea el prompt del sistema con información de productos y FAQs"""
    from products import products, faqs
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

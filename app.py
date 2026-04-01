import streamlit as st
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*Core Pydantic V1 functionality.*")

from src.bot import BetterChatbot
from src.settings import get_logger

# ---------------------------
# Configurações iniciais
# ---------------------------
st.set_page_config(
    page_title="DevoBot – Assistente Teológico",
    page_icon="📜",
    layout="centered"
)

load_dotenv()
logger = get_logger(__name__)

# Carrega o texto base (igual ao seu script original)
with open("assets/course_notes.txt", "r", encoding="utf-8") as f:
    course_text = f.read()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚙️ Menu")
st.sidebar.write("Gerencie sua sessão de conversa.")

if st.sidebar.button("🔄 Resetar Conversa"):
    st.session_state.messages = []
    st.sidebar.success("Conversa reiniciada!")
    st.rerun()

# ---------------------------
# Instanciar o bot (uma única vez)
# ---------------------------
if "bot" not in st.session_state:
    st.session_state.bot = BetterChatbot(
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
        course_text=course_text,
    )

# Histórico das mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Cabeçalho
# ---------------------------
st.title("📜 DevoBot – Assistente Teológico")
st.subheader("Seu apoio para estudos bíblicos e compreensão teológica.")


st.divider()

# ---------------------------
# Exibe histórico de mensagens
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------
# Entrada do usuário
# ---------------------------
user_input = st.chat_input("Digite sua pergunta teológica...")

if user_input:
    # Mostra a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Gera resposta do bot
    with st.chat_message("assistant"):
        with st.spinner("Consultando as Escrituras..."):
            try:
                response = st.session_state.bot.chat(user_input)
            except Exception as e:
                response = f"Erro ao processar a resposta: {e}"
                logger.error(e)
        st.write(response)

    # Salva mensagem do bot
    st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------------------
# Rodapé
# ---------------------------
st.divider()
st.caption("Desenvolvido com Streamlit — Por Amanda E.")
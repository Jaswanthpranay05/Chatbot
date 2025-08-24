import streamlit as st
from transformers import pipeline

# -------- Load Model --------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

chatbot = load_model()

# -------- Streamlit UI --------
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.title("💬 AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for user, bot in st.session_state.messages:
    st.chat_message("user").markdown(user)
    st.chat_message("assistant").markdown(bot)

# Input box
if user_input := st.chat_input("Type your message…"):
    st.chat_message("user").markdown(user_input)
    response = chatbot(user_input, max_new_tokens=200)[0]["generated_text"]
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append((user_input, response))


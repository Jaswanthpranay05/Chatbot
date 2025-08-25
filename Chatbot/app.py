import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Model selection UI
st.title("💬 AI Chatbot")
model_choice = st.selectbox("Choose a model:", [
    "microsoft/DialoGPT-medium",
    "facebook/blenderbot_small-90M"
])

tokenizer, model = load_model(model_choice)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You: ")

if user_input:
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    reply_ids = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    bot_reply = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_reply))

for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")

# chat_app.py
# This is the chat interface that talks to your FastAPI backend

import streamlit as st
import requests

# ── Page Configuration ───────────────────────────────────────────
st.set_page_config(page_title="MediAssist", page_icon="🩺")

st.title("🩺 MediAssist")
st.caption("Ask me anything about diabetes — symptoms, risk factors, prevention, and more.")

# ── The FastAPI Backend URL ──────────────────────────────────────
API_URL = "http://127.0.0.1:8000/ask"

# ── Keep Chat History Across Messages ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display Previous Messages ────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── Chat Input Box ────────────────────────────────────────────────
user_question = st.chat_input("Type your question here...")

if user_question:
    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Send the question to the FastAPI backend and get the answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": user_question},
                    timeout=900   # Allow up to 15 minutes for slower CPU responses
                )
                answer = response.json()["answer"]
            except Exception as e:
                answer = f"⚠️ Could not reach MediAssist API: {e}"

            st.write(answer)

    # Save the assistant's reply to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
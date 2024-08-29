import streamlit as st
from openai_helper import OpenAIChat


st.title("💬 Chatbot")




if 'chat' not in st.session_state:
    st.session_state.chat = OpenAIChat(openai_model='gpt-4o-mini')
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

uploaded_file = st.file_uploader("Choose a document to upload")
if uploaded_file is not None:
    print(type(uploaded_file))

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = st.session_state.chat.send_message(prompt)
    message = response.get('text')

    st.session_state.messages.append({"role": "assistant", "content": message})
    st.chat_message("assistant").write(message)



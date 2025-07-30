from env import *

st.set_page_config(
    page_title="Databricks Doc Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Documents", "Chatbot", "Clear Storage"])

if page == "Documents":
    from upload import upload
    upload.upload()
elif page == "Chatbot":
    from chatbot import chatbot
    chatbot.chatbot(persist_dir,embeddings)
elif page == "Clear Storage":
    from deletion import deletion_main
    deletion_main.delete(persist_dir,embeddings)
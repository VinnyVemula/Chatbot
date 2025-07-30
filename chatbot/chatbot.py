from env import *
from langchain.chains import RetrievalQA,LLMChain,StuffDocumentsChain
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from pathlib import Path
def vector_store_search(query,persist_dir,embedding_model):
    # Load your vector store (already normalized if using cosine similarity)
    vector_store = FAISS.load_local(persist_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    # retriever = vector_store.as_retriever()
    template = """
    You are an AI Agent. You are a professional document analyzer and answers the user 
    questions in a lucid and simple way to remeber for long time.
    Use the following context to answer the question.
    If the question is some kind of greeting then respond and if the answer is not in the context,
    say: "I don't have knowledge on this subject.
    {context}

    Question: {question}
    """
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=template
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain,
    document_variable_name="context")
    qa_chain = RetrievalQA(
        combine_documents_chain=stuff_chain,
        retriever=retriever,
        input_key="query",
        return_source_documents = True,
        verbose = True
        )
    return qa_chain.invoke({"query": query})
def chatbot(persist_dir,embedding_model):
    st.set_page_config(page_title="AI Document Chatbot", layout="centered")
    st.title("📄 AI Document Chatbot")

    # Initialize chat history in session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask something about the documents...")
    if user_input:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👨‍💻"):
            st.markdown(user_input)

        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                faiss_index_path = Path(persist_dir) / "index.faiss"
                if faiss_index_path.exists():
                    response = vector_store_search(user_input,persist_dir,embedding_model)
                    result = response["result"]
                    source_doc = str([ source.metadata['source'] for source in response['source_documents']])
                    st.markdown(result)
                    # if ("I don't have knowledge on the subject" not in result):
                    #     st.markdown(f"Sources to refer: {source_doc}")
                        # st.balloons()
                    st.session_state.chat_history.append({"role": "assistant", "content": result})
                else:
                    messages_container = st.container()
                    with messages_container:
                        result = "Hello, there are no documents to search, please upload documents...."
                        st.info(result)
                        st.session_state.chat_history.append({"role": "assistant", "content": result})
    if st.button("clear chat"):
        st.session_state.chat_history = []

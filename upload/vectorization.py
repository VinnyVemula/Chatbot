import tempfile
from env import *
from deletion import deletion
from pathlib import Path

# import faiss
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
# import langchain_community.vectorstores.FAISS
# from langchain.docstore import InMemoryDocstore


def extract_text_from_pdf(file) -> str:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_txt(file) -> str:
    with open(file, "r", encoding="utf-8") as f:
        return f.read()

    return file.read().decode("utf-8")
def process_uploaded_files(uploaded_files,persist_dir,embeddings) -> List[Document]:
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name in deletion.list_of_files(persist_dir,embeddings):
            continue
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            if uploaded_file.name.endswith(".pdf"):
                text = extract_text_from_pdf(temp_file_path)
            elif uploaded_file.name.endswith((".txt", ".md")):
                text = extract_text_from_txt(temp_file_path)
            else:
                print("This document contains unsupported file format {uploaded_file.name}")
                continue
            metadata = {
                "source": uploaded_file.name,
                "file_type": uploaded_file.type,
                "size": uploaded_file.size
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
        finally:
            os.unlink(temp_file_path)
    
    return documents

def split_documents(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> List[Document]:
    """
    Optimized document splitter using LangChain's built-in TokenTextSplitter.
    TokenTextSplitter is built on tiktoken.
    Args:
        documents: List of Langchain Document objects
        chunk_size: Target size in tokens (default: 512)
        chunk_overlap: Overlap in tokens (default: 64)
    
    Returns:
        List of chunked Documents with preserved metadata
    """
    token_splitter = TokenTextSplitter(
        encoding_name="cl100k_base",  # GPT-4/3.5/embedding tokenizer
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return token_splitter.split_documents(documents)
# def create_vector_store(docs: List[Document], embedding_model: str, persist_dir: str):
#     return FAISS.from_documents(docs, embedding_model,persist_dir)
def create_vector_store(docs: List[Document], embedding_model: str, persist_dir: str ) -> FAISS:
    """
    Creates a FAISS vector store with IVF index and persistent storage.
    
    Args:
        docs: List of Document objects
        embedding_model: Embedding model instance
        persist_dir: Directory to save the index
        n_clusters: Number of IVF clusters (default: 100)
        n_probe: Clusters to search at query time (default: 10)
    
    Returns:
        FAISS vector store with IVF index
    """
    # Create directory if it doesn't exist
    os.makedirs(persist_dir, exist_ok=True)
    # Extract embeddings
    # embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])
    # embeddings_array = np.array(embeddings).astype('float32')
    # dimension = embeddings_array.shape[1]
    # # Create IVF index
    # quantizer = FAISS.IndexFlatL2(dimension)  # How vectors are compared
    # n_clusters = int(np.sqrt(len(embeddings))) # Creates optimum number of vectors
    # index = FAISS.IndexIVFFlat(quantizer, dimension, n_clusters, FAISS.METRIC_L2)
    # index.nprobe = max(5, n_clusters//20) # 5-10% of n_clusters is optimal for searching the clusters
    # # Train the index (requires representative data)
    # index.train(embeddings_array)
    # # Add vectors to index
    # index.add(embeddings_array)
    # # Create LangChain FAISS wrapper
    # index_to_docstore_id = {str(i): str(i) for i in range(len(docs))}
    # docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
    # vector_store = FAISS(
    #     embedding_function=embedding_model.embed_query,
    #     index=index,
    #     docstore=docstore,
    #     index_to_docstore_id=index_to_docstore_id
    # )
    faiss_index_path = Path(persist_dir) / "index.faiss"
    if faiss_index_path.exists():
        vector_store = FAISS.load_local(persist_dir, embeddings=embedding_model,allow_dangerous_deserialization=True)
        new_vector_store = FAISS.from_documents(docs, embedding=embedding_model)
        vector_store.add_documents(new_vector_store.docstore._dict.values())  # 🔁 Append new docs to existing index
    else:
        vector_store = FAISS.from_documents(docs, embedding=embedding_model)
    # Save to disk
    vector_store.save_local(persist_dir)
    # st.session_state.some_flag = True
    # del st.session_state.some_flag
    # st.rerun()
    return vector_store

def available_documents(persist_dir,embeddings):
    messages_container = st.container()
    faiss_index_path = Path(persist_dir) / "index.faiss"
    if faiss_index_path.exists():
        files_list = deletion.list_of_files(persist_dir, embeddings)
        if files_list:
            if len(files_list) == 1 and "all" in files_list:
                with messages_container:
                    st.info("No documents to display, please upload documents")
            else:
                with st.container(height=300):  # Fixed height with scroll
                    for file in files_list:
                        if "all" in file:
                            continue
                        st.write(f"📄 {file}")
        else:
            with messages_container:
                st.info("No documents to display, please upload documents")
    else:
        with messages_container:
            st.info("No documents to display, please upload documents")
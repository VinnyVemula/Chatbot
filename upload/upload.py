# import env, vectorization
from env import *
from deletion import deletion
def upload():
    from . import vectorization
    st.set_page_config(
        page_title="Document Vectorizer", 
        page_icon="📄",
        layout="wide"
    )
    st.title("📄 Document Vectorizer")
    st.header("Upload documents to answer your queries")
    
    # File uploader section
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, Markdown"
    )
    if uploaded_files:
        with st.status("Processing documents...", expanded=True) as status:
            st.write("1. Extracting text from documents...")
            documents = vectorization.process_uploaded_files(uploaded_files,persist_dir,embeddings)
            if documents:
                st.write("2. Splitting the generated text...")
                split_docs = vectorization.split_documents(documents)
                st.write(f"3. Generating embeddings using {embeddings}...")
                vector_store = vectorization.create_vector_store(
                    split_docs,
                    embeddings,
                    persist_dir
                )
                
                status.update(label="Processing complete!", state="complete")
                st.success(f"Successfully processed {len(uploaded_files)} files with {len(split_docs)} chunks")
            else:
                status.update(label="Already Processed!", state="complete")
        st.header("Available documents")
    if st.button("Get files list"):
        vectorization.available_documents(persist_dir,embeddings)


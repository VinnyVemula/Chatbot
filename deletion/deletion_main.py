import streamlit as st
import os
import shutil
from langchain_community.vectorstores import FAISS
from . import deletion
from pathlib import Path
def delete(persist_dir: str, embeddings: str):
    from . import deletion
    st.set_page_config(
        page_title="File Deletion", 
        page_icon="📄",
        layout="wide"
    )
    st.title("📄 Document Deletion")
    
    # Clear previous messages container
    messages_container = st.container()
    
    faiss_index_path = Path(persist_dir) / "index.faiss"
    if faiss_index_path.exists():
        messages_container = st.container()
        files_list = deletion.list_of_files(persist_dir, embeddings)
        if files_list:
            with messages_container:
                st.info(f"Select the Document that you want to delete:")
            
            file_names = st.multiselect(
                "File Names:",
                deletion.list_of_files(persist_dir, embeddings)
            )
            
            if file_names:  # Only show messages if files are selected
                with messages_container:
                    if "all" in file_names:
                        shutil.rmtree(persist_dir)
                        st.info("Deleted all files")
                    else: 
                        for file_name in file_names:
                            status = deletion.file_deletion(file_name, persist_dir, embeddings)
                            st.info(status)
        else:
            with messages_container:
                st.info("No files found in directory")
    else:
        with messages_container:
            st.info("No documents to display, please upload documents")
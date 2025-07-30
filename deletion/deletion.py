from langchain_community.vectorstores import FAISS
import shutil
import os
import streamlit as st
from pathlib import Path
def list_of_files(persist_dir,embeddings):
    faiss_index_path = Path(persist_dir) / "index.faiss"
    file_set = set()
    if faiss_index_path.exists():
        data = FAISS.load_local(persist_dir, embeddings=embeddings,allow_dangerous_deserialization=True)
        docstore_dict = data.docstore._dict
        for doc_id, doc in docstore_dict.items():
            file_name = doc.metadata.get("source") or doc.metadata.get("file_name") or "Unknown"
            file_set.add(file_name)
    file_list = list(file_set)
    file_list.insert(0, "all")
    return file_list
def file_deletion(file_to_delete,persist_dir,embeddings):
    data = FAISS.load_local(persist_dir, embeddings,allow_dangerous_deserialization=True)
    # Step 1: Find doc IDs for this file
    doc_ids_to_delete = [
        doc_id for doc_id, doc in data.docstore._dict.items()
        if doc.metadata.get("source") in file_to_delete
    ]

    # Step 2: Delete from vector store
    if doc_ids_to_delete:
        data.delete(doc_ids_to_delete)
        data.save_local(persist_dir)
        return f"Deleted {len(doc_ids_to_delete)} documents from {file_to_delete}"
    else:
        return f"No documents found for {file_to_delete}"

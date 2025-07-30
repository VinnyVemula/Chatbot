import streamlit as st
from PyPDF2 import PdfReader
import tiktoken
from databricks_langchain import ChatDatabricks
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from typing import List
from langchain_community.vectorstores import FAISS
import os
from databricks_langchain import DatabricksEmbeddings
import numpy as np
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0.2
)
embeddings = DatabricksEmbeddings(
    endpoint="databricks-bge-large-en"
)
persist_dir = "./storage/faiss"
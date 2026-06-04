import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

UPLOAD_DIR = "uploaded_docs"
CHROMA_DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small", model_kwargs={'device': 'cpu'})
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

def process_and_add_document(file_path, document_id):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        return False

    documents = loader.load()

    for doc in documents:
        doc.metadata["document_id"] = document_id

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)

    vectorstore.add_documents(documents=splits)
    return True

def get_retriever_by_doc_id(document_id, k=3):
    return vectorstore.as_retriever(
        search_kwargs={'filter': {'document_id': document_id}, 'k': k}
    )
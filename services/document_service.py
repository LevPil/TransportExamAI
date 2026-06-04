import os
from database import Document
from rag_pipeline import UPLOAD_DIR, process_and_add_document

def get_all_documents(db):
    return db.query(Document).all()

def save_uploaded_document(db, uploaded_file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    new_doc = Document(filename=uploaded_file.name, filepath=file_path)
    db.add(new_doc)
    db.commit()
    success = process_and_add_document(file_path, new_doc.id)
    return new_doc, success
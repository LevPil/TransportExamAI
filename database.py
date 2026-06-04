import os
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "postgresql://admin:adminpassword@localhost:5432/transport_exam"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    filepath = Column(String) # Путь к файлу на диске
    upload_time = Column(DateTime, default=datetime.utcnow)

    tests = relationship("Test", back_populates="document")

class Test(Base):
    __tablename__ = "tests"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    content = Column(Text)
    correct_answers = Column(Text)
    document_id = Column(Integer, ForeignKey("documents.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="tests")
    results = relationship("TestResult", back_populates="test")

class TestResult(Base):
    __tablename__ = "test_results"
    id = Column(Integer, primary_key=True, index=True)
    student_name = Column(String)
    test_id = Column(Integer, ForeignKey("tests.id"))
    score = Column(String)
    completed_at = Column(DateTime, default=datetime.utcnow)

    test = relationship("Test", back_populates="results")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    print("✅ База данных успешно инициализирована и таблицы созданы!")
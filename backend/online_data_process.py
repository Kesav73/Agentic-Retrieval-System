from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

db = None

def create_retriever_from_pdf(filepath, embeddings=None):
    """Creates a retriever tool from a PDF file."""
    embeddings = embeddings or OpenAIEmbeddings()
    try:
        # 1. Load the PDF file
        loader = PyPDFLoader(filepath)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF file: {e}")
        return None

    try:
        # 2. Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error splitting text: {e}")
        return None

    try:
        # 3. Create a FAISS vector store from the chunks and embeddings
        global db
        if not db:
            db = FAISS.from_documents(docs, embeddings)                    
        else:
            db.add_documents(docs)
            
        db.save_local("./vector_db")
        retriever = db.as_retriever()

    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return None

    return retriever

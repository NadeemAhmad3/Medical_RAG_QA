import pandas as pd
import pickle
import os
import faiss
import time
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

from src.config import DATA_PATH, VECTOR_DB_PATH, COHERE_API_KEY, EMBEDDING_MODEL

# We need a place to save the "Parents" (The full text)
PARENT_DOC_STORE_PATH = "vectorstore/parent_docs"

def create_retriever_and_db():
    # 1. Load Data
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH).fillna("")
    df = df.head(4000)  # Limit to 4000 documents
    
    docs = []
    for _, row in df.iterrows():
        # The 'Parent' is the full transcription
        content = row['transcription']
        
        # Metadata allows us to filter later if needed
        metadata = {
            "specialty": row['medical_specialty'],
            "sample_name": row['sample_name'],
            "keywords": row['keywords']
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # 2. Define Splitters
    # Parent splitter: Cuts the document into large distinct sections
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    # Child splitter: Cuts those sections into tiny search snippets
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # 3. Setup Storage
    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model=EMBEDDING_MODEL)
    
    # The Vector Store holds the CHILDREN (for searching)
    # Initialize empty FAISS index with correct embedding dimension (1024 for embed-english-v3.0)
    embedding_dim = 1024
    index = faiss.IndexFlatL2(embedding_dim)
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    # The Doc Store holds the PARENTS (for reading)
    # We use InMemoryStore for the parent documents
    store = InMemoryStore()

    # 4. The Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print("Ingesting documents (This handles Parent-Child splitting)...")
    # Process in batches to avoid rate limits (Trial API: 100 calls/min)
    batch_size = 10
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1} ({len(batch)} docs)...")
        retriever.add_documents(batch, ids=None)
        if i + batch_size < len(docs):
            time.sleep(2)  # Small delay between batches to avoid rate limit

    # 5. Saving (Crucial Step)
    # FAISS only saves the vectors (children). We must manually save the DocStore (parents).
    vectorstore.save_local(VECTOR_DB_PATH)
    
    # Save the parent documents using pickle
    if not os.path.exists("vectorstore"): os.makedirs("vectorstore")
    with open(f"{VECTOR_DB_PATH}/docstore.pkl", "wb") as f:
        pickle.dump(store, f)
        
    print(f"Database saved. {len(docs)} parent documents processed.")

if __name__ == "__main__":
    create_retriever_and_db()
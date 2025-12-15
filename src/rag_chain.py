import pickle
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings, ChatCohere, CohereRerank
from langchain_classic.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import VECTOR_DB_PATH, COHERE_API_KEY, EMBEDDING_MODEL, RERANK_MODEL, CHAT_MODEL

def get_qa_chain():
    # 1. Load Embeddings
    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model=EMBEDDING_MODEL)

    # 2. Load VectorStore (Children)
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    # 3. Load DocStore (Parents)
    with open(f"{VECTOR_DB_PATH}/docstore.pkl", "rb") as f:
        docstore = pickle.load(f)

    # 4. Reconstruct the Parent-Child Retriever
    # This retriever searches Children (vectors) but returns Parents (text)
    base_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
    )

    # 5. Apply Cohere Rerank
    # We retrieve parents first, then Rerank the parents to ensure they are perfect.
    compressor = CohereRerank(
        cohere_api_key=COHERE_API_KEY, 
        model=RERANK_MODEL, 
        top_n=3
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    # 6. Chat Model & Prompt (Same as before)
    llm = ChatCohere(cohere_api_key=COHERE_API_KEY, model=CHAT_MODEL, temperature=0)
    
    prompt_template = """
    You are an expert medical assistant. Answer based on the context provided.
    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain
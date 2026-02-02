
           
           
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_faiss(texts):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(
        texts=texts,
        embedding=embeddings
    )

def retrieve_similar_documents(faiss_index, query, k=4):
    return faiss_index.similarity_search(query, k=k)
           
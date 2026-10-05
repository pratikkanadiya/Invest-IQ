import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

load_dotenv()

embedding_model = os.getenv("MODEL_NAME") or "BAAI/bge-small-en-v1.5"

hf_model = SentenceTransformer(embedding_model)

class SafeLangChainEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

langchain_embeddings = SafeLangChainEmbeddings(hf_model)
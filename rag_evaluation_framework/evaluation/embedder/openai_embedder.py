from rag_evaluation_framework.evaluation.embedder.base import Embedder
from langchain_openai import OpenAIEmbeddings

class OpenAIEmbedder(Embedder):

    model_name: str

    def __init__(self, model_name: str="text-embedding-3-small"):
        self.model_name = model_name

    def embed_docs(self, docs: list[str]) -> list[list[float]]:
        return OpenAIEmbeddings(model=self.model_name).embed_documents(docs)

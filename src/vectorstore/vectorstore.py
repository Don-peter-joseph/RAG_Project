from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever,EnsembleRetriever

class VectorStore:
    def __init__(self):
        self.embedding=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        self.vectorstore=None
        self.retriever=None
        self.bm25retriever=None
        self.hybrid_retriever=None
    
    def create_retriever(self,documents:List[Document]):
        """creates vectorstore from documents
        Args:
            documents: documents to be be embedded
        """
        self.vectorstore=FAISS.from_documents(
            documents=documents,
            embedding=self.embedding
        )
        self.retriever=self.vectorstore.as_retriever()

    def create_bm25_retriever(self,documents:List[Document]):
        sparse_retriever=BM25Retriever.from_documents(documents)
        sparse_retriever.k=3
        self.bm25retriever=sparse_retriever

    def create_hybrid_retriever(self,documents:List[Document]):
        if not self.retriever:
            self.create_retriever(documents)
        if not self.bm25retriever:
            self.create_bm25_retriever(documents)
        self.hybrid_retriever=EnsembleRetriever(
            retrievers=[self.retriever,self.bm25retriever],
            weights=[0.6,0.4]
        )
    

    def get_retriever(self):
        """get the retriever
        Returns: 
            retriever instance"""
        if not self.retriever:
            raise ValueError('Vectorstore not initilalized. call create_retriever method first')
        return self.hybrid_retriever
    
    def retrieve(self,query:str,k:int=4)->List[Document]:
        """retrieves the relevent documents of query from vectorstore
        
        Args:
            query: user input
            k : no of document to return 
        return :
            List of retrieved documents"""
        if not self.retriever:
            raise ValueError('Vectorstore not initilalized. call create_retriever method first')
        return self.retriever.invoke(query)
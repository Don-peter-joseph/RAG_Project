from typing import List,Union
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from pathlib import Path

class DocumentProcessor:
    "handle document loading and processing"

    def __init__(self,chunk_size:int=500, chunk_overlap:int=50):
        """Initialize
        Args:
            chunk_size: size of text chunks
            chunk_overlap: overlap between chunks"""
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self,url:str)->List[Document]:
        """load document from all urls"""
        loader=WebBaseLoader(url)
        return loader.load()
    
    def load_from_txt(self,file_path: Union[str,Path])->List[Document]:
        """load document from text file"""
        loader=TextLoader(str(file_path),encoding=('utf-8'))
        return loader.load()
    
    def load_from_pdf(self,file_path:Union[str,Path])->List[Document]:
        """Load document from pdf"""
        loader=PyPDFLoader(file_path)
        return loader.load()
    
    def load_from_pdf_directory(self,directory:Union[str,Path])->List[Document]:
        """Load document from pdf directory"""
        loader=PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_documents(self,sources=List[str])->List[Document]:
        """Load documents from anywhere
        Args:
            sources :List of urls, paths
        Returns:
            list of loaded documents"""
        docs:List[Document]=[]
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
            path=Path(src)
            if path.is_dir():
                docs.extend(self.load_from_pdf_directory(path))
            elif path.suffix.lower()=='.txt':
                docs.extend(self.load_from_txt(src))
            elif path.suffix.lower()=='.pdf':
                docs.extend(self.load_from_pdf(src))
            else:
                raise ValueError(
                    f'unsupported source type {src}.'
                    'use url, text or pdf.'
                )
        return docs
    
    def split_documents(self,documents:List[Document])->List[Document]:
        """Split documents into chunks
        Args:
            documents: List of documents to split
        Return:
            List of split documents"""
        return self.text_splitter.split_documents(documents)
    
    def process_documents(self,urls:List[str])->List[Document]:
        """
        Pipeline to load and split documents
        Args:
            urls: list of urls to process
        Return :
            list of processed document chunks"""
        docs=self.load_from_documents(sources=urls)
        return self.split_documents(docs)
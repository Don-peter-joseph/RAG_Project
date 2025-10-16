from pydantic import BaseModel,Field
from langchain.schema import Document
from typing import List

class GraphState(BaseModel):
    question:str
    retrieved_docs: List[Document] = []
    answer:str=""
from typing import List,Optional
from src.state.graph_state import GraphState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

class AgentNode:
    def __init__(self,retriever,llm):
        self.retriever=retriever
        self.llm=llm
        self.agent=None

    def retrieve_docs(self,state:GraphState)->GraphState:
        docs=self.retriever.invoke(state.question)
        return GraphState(
            question=state.question,
            retrieved_docs=docs
        )

    def init_tools(self)->List[Tool]:
        "retriever and wikipedea tool"
        def retriever_tool_fn(query:str)->str:
            docs:List[Document]=self.retriever.invoke(query)
            if not docs:
                return 'no documents found'
            merged=[]
            for i,d in enumerate(docs[:8],start=1):
                meta=d.metadata if hasattr(d,"metadata") else {}
                title=meta.get('title') or meta.get("source") or f'doc_{i}'
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)
        retriever_tool=Tool(
            name='retriever',
            description='fetch passages from indexed vectorstore',
            func=retriever_tool_fn
        )

        wiki=WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3,lang='en')
        )
        wikipedia_tol=Tool(
            name='wikipedia',
            description='search in wiki',
            func=wiki.run
        )

        return [retriever_tool,wikipedia_tol]

    def agent_build(self):
        """react agent with tools"""
        tools=self.init_tools()
        system_prompt=(
            "You are a helpful RAG agent"
            "Prefer 'retriever' for the question. If answer not found , use wikipedia"
            "Return only the final useful answer"
        )
        self.agent=create_react_agent(self.llm,tools=tools,prompt=system_prompt)

    def generate_answer(self,state:GraphState)->GraphState:
        if self.agent is None:
            self.agent_build()

        result=self.agent.invoke({'messages':[HumanMessage(content=state.question)]})
        messages=result.get('messages',[])
        ans :Optional[str]= None

        if messages:
            answer_msg=messages[-1]
            ans=getattr(answer_msg,"content",None)
    
        return GraphState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=ans or "couldn't generate answer"
        )
    
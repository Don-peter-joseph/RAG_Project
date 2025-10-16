from langgraph.graph import StateGraph, END
from src.state.graph_state import GraphState
from src.nodes.agent_node import AgentNode

class GraphBuilder:
    def __init__(self,retriever,llm):
        self.nodes=AgentNode(retriever,llm)
        self.graph=None
    
    def build(self):
        flow=StateGraph(GraphState)

        flow.add_node('retriever',self.nodes.retrieve_docs)
        flow.add_node('generator',self.nodes.generate_answer)

        flow.set_entry_point('retriever')

        flow.add_edge('retriever','generator')
        flow.add_edge('generator',END)

        self.graph=flow.compile()
        return self.graph
    

    def run(self,question:str)->dict:
        if not self.graph:
            self.build()
        initial_state=GraphState(question=question)
        return self.graph.invoke(initial_state)
        
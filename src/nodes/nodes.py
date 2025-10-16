from src.state.graph_state import GraphState

class RagNodes:
    def __init__(self,retriever,llm):
        self.retriever=retriever
        self.llm=llm

    def retrieve_docs(self,state:GraphState)->GraphState:
        docs=self.retriever.invoke(state.question)
        return GraphState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def generate_answer(self,state:GraphState)->GraphState:
        context="\n\n".join([doc.page_content for doc in state.retrieved_docs])
        prompt= f"""answer the question based on the context.
        {context}
        Question: {state.question}
        """
        response= self.llm.invoke(prompt)
        return GraphState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )
    

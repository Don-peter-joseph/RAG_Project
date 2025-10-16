# RAG_Project
Simple Rag project that takes pdfs or text as input  and responds to the user based on the documents. If not found uses wikipedia.

# Usage
 1. Clone the repository to the local.
 2. Change to the Rag_Project directory.
 3. Get the api key for gemini from google cloud console, api key for huggingface and langsmith credentials.
 4. Create a new file .env and define the secrets as below:
        ```huggingface_token = ""
        LANGSMITH_TRACING='true'
        LANGSMITH_ENDPOINT='https://api.smith.langchain.com'
        LANGSMITH_API_KEY=''
        LANGSMITH_PROJECT=''
        GEMINI_API_KEY='''```
 5. Install the libraries from requirement.txt: `uv add -r requirements.txt`. You have to install uv if you don't have or use pip or another package manager.
 6. Run the command : `streamlit run streamlit_app.py` in the terminal

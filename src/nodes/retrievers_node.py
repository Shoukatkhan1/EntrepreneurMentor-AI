
from ingest import docsearch
from langchain_core.tools import tool


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# 🔍 Retriever Tool using LangGraph and retrieve answers from the knowledge base
@tool  
def retriever_tool(query: str) -> str:  
    """Searches and returns information from the startup knowledge base about funding, growth, marketing, team building, and product-market fit."""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "NO_RESULTS_FOUND"
        
        results = [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        return "\n\n".join(results)
    except Exception as e:
        return f"ERROR: {str(e)}"
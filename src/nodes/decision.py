# ===========================================================
# decision.py - Decision functions for LangGraph routing
# Author: Shoukat Khan
# ===========================================================

from src.nodes.schemas import AgentState
from langchain_core.messages import AIMessage, HumanMessage


def should_use_tool(state: AgentState) -> bool:
    """
    Determine if tools should be called based on the last message.
    
    Args:
        state: Current agent state containing messages
        
    Returns:
        bool: True if LLM wants to use tools, False otherwise
    """
    messages = state.get("messages", [])
    
    # If no messages, no tools needed
    if not messages:
        return False
    
    last_message = messages[-1]
    
    # Check if last message is from AI and has tool calls
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return True
    
    return False


def should_summarize(state: AgentState) -> str:
    """
    Decide whether to summarize the conversation or continue to LLM.
    
    Summarization happens when conversation exceeds threshold to prevent
    context overflow and maintain performance.
    
    Args:
        state: Current agent state containing messages
        
    Returns:
        str: "summarize" if threshold exceeded, "llm" to continue normally
    """
    messages = state.get("messages", [])
    
    # Count only user messages (HumanMessage) for threshold
    # Tool messages and AI messages don't count toward conversation length
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    
    # Summarize if more than 4 user messages
    # This keeps context manageable while preserving conversation flow
    if len(user_messages) > 4:
        return "summarize"
    
    # Continue to LLM without summarization
    return "llm"
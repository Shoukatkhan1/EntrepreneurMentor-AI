from typing import Sequence, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from operator import add as add_messages


class AgentState(TypedDict):
    """
    State schema for the RAG agent.
    
    Attributes:
        messages: The conversation messages (with add_messages reducer)
        summary: Optional summary of older conversation
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str
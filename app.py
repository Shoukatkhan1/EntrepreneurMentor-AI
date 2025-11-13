# ===========================================================
# 💼 EntrepreneurMentor AI — DB-Only Startup Assistant (Streamlit)
# Latest LangGraph Syntax | Clean | Recursion-Safe | With ToolNode
# Author: Shoukat Khan
# ===========================================================

import streamlit as st
from dotenv import load_dotenv
import os
import uuid
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode  #  Import ToolNode
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# Import custom modules
from prompt import system_prompt
from src.nodes.schemas import AgentState
from src.nodes.decision import should_use_tool, should_summarize
from src.nodes.retrievers_node import retriever_tool
from src.nodes.memory import checkpointer

# ----------------------------------------------------------
# 🌿 Load Environment Variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY= os.getenv("TAVILY_API_KEY", "")

# ----------------------------------------------------------
memory = checkpointer  # Use the checkpointer from memory.py

# ----------------------------------------------------------
# 🌐 Streamlit UI Setup
st.set_page_config(page_title="EntrepreneurMentor AI", layout="wide", page_icon="💼")

# Header
st.markdown("<h1>💼 EntrepreneurMentor AI Assistant</h1>", unsafe_allow_html=True)
st.caption("Your startup mentor — answers strictly from DB content and web (funding, growth, marketing, team, PMF).")
st.caption("Powered by LangGraph, Groq LLM, Pinecone, and Tavily Search.")
# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")
st.sidebar.markdown("**👨‍💻 Developed by:** Shoukat Khan")
st.sidebar.markdown("**💼 Project:** EntrepreneurMentor AI")
st.sidebar.markdown("### 🎯 Features")
st.sidebar.markdown("""- 🧠 Smart summarization (6+ messages)
- 🧠 Persistent memory with Neon DB
- 🔍 RAG-powered retrieval (LANGRAPH)
- ⚡ Optimized context handling
- 🤖 Groq LLM integration
- 🛠️ ToolNode for automatic tool execution
""")
st.sidebar.markdown("---")

# Sidebar Controls for Model and Temperature
model_choice = st.sidebar.selectbox("Choose Model:", ["openai/gpt-oss-120b","llama-3.3-70b-versatile","llama-3.1-8b-instant"])
temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.3, 0.05)

# Clear Chat Button for clearing session
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

st.sidebar.markdown("---")
if "chat_history" in st.session_state:
    st.sidebar.write(f"💬 **Messages:** {len(st.session_state.chat_history)}")

# ----------------------------------------------------------
# 🔹 Model Initialization 
llm = ChatGroq( 
    model=model_choice,
    temperature=temperature,
    api_key=GROQ_API_KEY
)

# ----------------------------------------------------------
# Tavily Search Tool
tavily_search = TavilySearchResults(max_results=3)

#  Bind Tools
tools = [retriever_tool, tavily_search]  # List of available tools
llm_with_tools = llm.bind_tools(tools)  # Bind tools to LLM mean connections tools with LLM

#  Create ToolNode - handles all tool execution automatically
tool_node = ToolNode(tools)

# ----------------------------------------------------------
# 🤖 LLM Node used for processing messages with optional summary when conversation gets long 
def call_llm(state: AgentState) -> AgentState:
    """Main LLM node that processes messages with optional summary."""
    summary = state.get("summary", "")  # Get existing summary if any
    previous_messages = state.get("messages", [])  # Get previous messages
    
    # Ensure messages is a list
    if not isinstance(previous_messages, list):   # Ensure previous_messages is a list
        previous_messages = [previous_messages]  # provide a list if it's not
    
    # Build system message with instructions
    system_content = system_prompt  # Base system prompt which created in prompt.py and provide instructions to the model
    
    if summary:
        system_content += f"\n\nSummary of earlier conversation: {summary}"  # Append summary if exists
    
    system_message = SystemMessage(content=system_content)  # Create system message
    
    # Include system message + all previous messages
    messages = [system_message] + previous_messages  # Combine system message with previous messages
    
    response = llm_with_tools.invoke(messages)  # Invoke LLM with tools
    
    return {"messages": [response]}  # Return the LLM response as the new messages

# ----------------------------------------------------------
# Summarization Node for conversation and summary for long chats
def summarize_conversation(state: AgentState) -> AgentState:
    """
    Summarizes conversation when it gets too long.
    Uses RemoveMessage to delete old messages while keeping recent context.
    """
    summary = state.get("summary", "")  # Get existing summary if any
    
    # Create summarization prompt
    if summary:
        summary_message = (    
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    # Add prompt to history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    # Delete all 
    # Keep last 4 messages for context
    messages_to_keep = state["messages"][-4:]
    
    return {"summary": response.content, "messages": messages_to_keep}

# ----------------------------------------------------------
# 🔗 LangGraph Definition with ToolNode
graph_builder = StateGraph(AgentState)  # Create a state graph for the agent

# Add nodes
graph_builder.add_node("llm", call_llm)   # LLM processing node
graph_builder.add_node("tools", tool_node)  # 🔥 ToolNode - automatic tool execution
graph_builder.add_node("summarize", summarize_conversation)  # Summarization node used for long conversations

# Set entry point
graph_builder.set_entry_point("llm")  # Set the entry point of the graph to the LLM node

# Define edges for conditional flows which helps in decision making for tool usage
# Add conditional edge for tool usage
graph_builder.add_conditional_edges( 
    "llm",
    should_use_tool,
    {
        True: "tools",  # used  ToolNode instead of custom function
        False: END
    }
)

# Add conditional edge for summarization check is conversation too long for summarization when exceeds 6 messages otherwise continue to LLM not summarize
graph_builder.add_conditional_edges(
    "tools",  # After tool execution, check if summarization is needed or not
    should_summarize,
    {
        "summarize": "summarize", # Summarize if needed 
        "llm": "llm" # Continue to LLM if no summarization needed
    }
)

# After summarization, go back to LLM
graph_builder.add_edge("summarize", "llm")  # Continue to LLM after summarization

# Compile the graph
rag_agent = graph_builder.compile(checkpointer=memory)  # Compile the graph with memory checkpointer used for saving state

# ----------------------------------------------------------
# 💬 Streamlit Session State
if "thread_id" not in st.session_state:  # Unique thread ID for session
    st.session_state.thread_id = str(uuid.uuid4())  # Generate a unique thread ID

if "chat_history" not in st.session_state:  # Initialize chat history
    st.session_state.chat_history = []  # List to store chat messages

config = {"configurable": {"thread_id": st.session_state.thread_id}}  # Configuration for agent with thread ID used for memory management for different sessions and users

# ----------------------------------------------------------
# 💬 Streamlit Chat Interface
user_query = st.chat_input("Ask your startup question...")  # Input box for user query

if user_query:
    # Add user message to history
    st.session_state.chat_history.append(("user", user_query))  # Append user query to chat history
    
    # Show thinking indicator
    with st.spinner("🤔 Thinking..."):
        try:
            # ✅ BUILD CONVERSATION CONTEXT FROM HISTORY
            conversation_messages = []
            
            # Get last 6 exchanges (12 messages: 6 user + 6 AI) to maintain context
            history_to_include = st.session_state.chat_history[-12:]
            
            # Convert session history to LangChain messages (exclude current query)
            for role, content in history_to_include[:-1]:  # [:-1] excludes current turn
                if role == "user":
                    conversation_messages.append(HumanMessage(content=content))
                else:
                    conversation_messages.append(AIMessage(content=content))
            
            # Add current query at the end
            conversation_messages.append(HumanMessage(content=user_query))
            
            # Invoke agent WITH FULL CONVERSATION CONTEXT
            result = rag_agent.invoke({"messages": conversation_messages}, config=config)
            
            # Extract AI response
            final_messages = result.get("messages", [])
            
            # Find the last AI message
            ai_response = None
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_response = msg.content
                    break
            
            if not ai_response:
                ai_response = "⚠️ No response generated. Please try again."
            
            # Add AI response to history
            st.session_state.chat_history.append(("assistant", ai_response))
        
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            if "413" in str(e) or "rate_limit" in str(e) or "too large" in str(e):
                error_msg = "⚠️ Conversation too long. Please click 'Clear Chat History' in sidebar."
            st.session_state.chat_history.append(("assistant", error_msg))

# ----------------------------------------------------------
# 🪄 Render Chat Messages
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"<div style='background:linear-gradient(90deg,#00c6ff,#0072ff);"
            f"color:#fff;padding:12px 20px;border-radius:25px;"
            f"max-width:70%;margin-bottom:10px'>{msg}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background:linear-gradient(90deg,#fbc2eb,#a6c1ee);"
            f"color:#311b92;padding:12px 20px;border-radius:25px;"
            f"max-width:70%;margin-left:auto;margin-bottom:10px'>{msg}</div>",
            unsafe_allow_html=True
        )
# ----------------------------------------------------------
# 🔹 Footer
st.markdown("""
<div style="text-align:center;font-size:0.9rem;color:#555;margin-top:30px;">
Developed by <strong>Shoukat Khan</strong> • © 2025
</div>
""", unsafe_allow_html=True)
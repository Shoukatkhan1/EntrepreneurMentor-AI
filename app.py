# ===========================================================
# 💼 EntrepreneurMentor AI — LangGraph 1.0.0
# Author: Shoukat Khan
# ===========================================================

import streamlit as st
from dotenv import load_dotenv
import os
import uuid
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver  
# Import custom modules
from prompt import system_prompt
from src.nodes.schemas import AgentState
from src.nodes.decision import should_use_tool, should_summarize
from src.nodes.retrievers_node import retriever_tool

# ----------------------------------------------------------
# 🌿 Load Environment Variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

memory = MemorySaver()  # ✅ Changed from InMemorySaver

# ----------------------------------------------------------
# 🌐 Streamlit UI Setup
st.set_page_config(page_title="EntrepreneurMentor AI", layout="wide", page_icon="💼")

st.markdown("<h1>💼 EntrepreneurMentor AI Assistant</h1>", unsafe_allow_html=True)
st.caption("Your startup mentor — answers from DB and web (funding, growth, marketing, team, PMF).")
st.caption("Powered by LangGraph 1.0, Groq LLM, Pinecone, and Tavily Search.")

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")
st.sidebar.markdown("**👨‍💻 Developed by:** Shoukat Khan")
st.sidebar.markdown("**💼 Project:** EntrepreneurMentor AI")
st.sidebar.markdown("### 🎯 Features")
st.sidebar.markdown("""
- 🧠 Smart summarization (6+ messages)
- 💾 Session memory management
- 🔍 RAG-powered retrieval
- 🌐 Web search integration (Tavily)
- ⚡ Optimized context handling
- 🤖 Groq LLM integration
- 🛠️ Custom tool execution (LangGraph 1.0)
""")
st.sidebar.markdown("---")

# Model Selection
model_choice = st.sidebar.selectbox(
    "Choose Model:", 
    ["llama-3.1-8b-instant","llama-3.3-70b-versatile""openai/gpt-oss-20b"]
)
temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.3, 0.05)

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

st.sidebar.markdown("---")
if "chat_history" in st.session_state:
    message_count = len(st.session_state.chat_history)
    st.sidebar.metric("💬 Messages", message_count)
    if message_count > 12:
        st.sidebar.warning("⚠️ Long conversation - consider clearing")

# ----------------------------------------------------------
# 🔹 Model Initialization
llm = ChatGroq(
    model=model_choice,
    temperature=temperature,
    api_key=GROQ_API_KEY
)

# ----------------------------------------------------------
# 🛠️ Setup Tools
tavily_search = TavilySearchResults(max_results=3)
tools = [retriever_tool, tavily_search]
llm_with_tools = llm.bind_tools(tools)

# Create tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools}

# ----------------------------------------------------------
# 🤖 LLM Node
def call_llm(state: AgentState) -> AgentState:
    """Main LLM node that processes messages with optional summary."""
    summary = state.get("summary", "")
    previous_messages = state.get("messages", [])
    
    # Ensure messages is a list
    if not isinstance(previous_messages, list):
        previous_messages = [previous_messages]
    
    # Build system message
    system_content = system_prompt
    
    if summary:
        system_content += f"\n\nSummary of earlier conversation: {summary}"
    
    system_message = SystemMessage(content=system_content)
    messages = [system_message] + previous_messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ----------------------------------------------------------
# 🛠️ Tool Execution Node (LangGraph 1.0 Compatible)
def execute_tools(state: AgentState) -> AgentState:

    last_message = state["messages"][-1]
    
    # Safety check
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}
    
    tool_calls = last_message.tool_calls
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        # Find and execute the tool
        tool_fn = tools_dict.get(tool_name)
        
        if not tool_fn:
            result = f"⚠️ Tool '{tool_name}' not found."
        else:
            try:
                # Execute the tool with its arguments
                result = tool_fn.invoke(tool_args)
            except Exception as e:
                result = f"⚠️ Error executing {tool_name}: {str(e)}"
        
        # Create ToolMessage
        results.append(
            ToolMessage(
                tool_call_id=tool_id,
                name=tool_name,
                content=str(result)
            )
        )
    
    return {"messages": results}

# ----------------------------------------------------------
# 📝 Summarization Node
def summarize_conversation(state: AgentState) -> AgentState:
    """Summarizes conversation when it gets too long."""
    summary = state.get("summary", "")
    
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    # Keep last 4 messages for context
    messages_to_keep = state["messages"][-4:]
    
    return {"summary": response.content, "messages": messages_to_keep}

# ----------------------------------------------------------
# 🔗 Build LangGraph
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", execute_tools)
graph_builder.add_node("summarize", summarize_conversation)

# Set entry point
graph_builder.set_entry_point("llm")

# Define edges
graph_builder.add_conditional_edges(
    "llm",
    should_use_tool,
    {
        True: "tools",
        False: END
    }
)

graph_builder.add_conditional_edges(
    "tools",
    should_summarize,
    {
        "summarize": "summarize",
        "llm": "llm"
    }
)

graph_builder.add_edge("summarize", "llm")

# Compile
rag_agent = graph_builder.compile(checkpointer=memory)

# ----------------------------------------------------------
# 💬 Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ----------------------------------------------------------
# 💬 Chat Interface
user_query = st.chat_input("Ask your startup question...")

if user_query:
    # Add user message to history
    st.session_state.chat_history.append(("user", user_query))
    
    # Show thinking indicator
    with st.spinner("🤔 Thinking..."):
        try:
            # ✅ BUILD CONVERSATION CONTEXT
            conversation_messages = []
            
            # Get last 10 messages (5 exchanges) for context
            history_to_include = st.session_state.chat_history[-6:]
            
            # Convert to LangChain messages
            for role, content in history_to_include:
                if role == "user":
                    conversation_messages.append(HumanMessage(content=content))
                else:
                    conversation_messages.append(AIMessage(content=content))
            
            # Add current query
            conversation_messages.append(HumanMessage(content=user_query))
            
            # Invoke agent with full conversation context
            result = rag_agent.invoke(
                {"messages": conversation_messages}, 
                config=config
            )
            
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
            
            # Handle specific errors
            if any(err in str(e).lower() for err in ["413", "rate_limit", "too large", "context"]):
                error_msg = "⚠️ Conversation too long. Please click 'Clear Chat History' in sidebar."
            
            st.session_state.chat_history.append(("assistant", error_msg))
            st.error(error_msg)

# ----------------------------------------------------------
# 🪄 Render Chat Messages with Gradient Style
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

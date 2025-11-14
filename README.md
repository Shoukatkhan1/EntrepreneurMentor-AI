💼 EntrepreneurMentor AI — LangGraph 1.0.0

Author: Shoukat Khan
Version: 1.0.0
License: MIT

Your AI-powered startup mentor: answers startup-related questions leveraging RAG, web search, and advanced LLMs.

🚀 Overview

EntrepreneurMentor AI is a cutting-edge AI assistant built for startup founders, entrepreneurs, and business enthusiasts. It provides real-time advice, summarization, and research-based insights using LangGraph, LangChain, Groq LLM, Pinecone, and Tavily Search.

The assistant combines RAG (Retrieval-Augmented Generation) with web search integration to answer questions about:

Startup growth & scaling strategies

Funding & investor guidance

Team building & product-market fit (PMF)

Marketing & go-to-market strategies

It also includes session memory and dynamic summarization, ensuring intelligent context-aware responses even during long conversations.

📦 Features

🧠 Contextual Summarization – Dynamically summarizes past conversations for coherent responses.

💾 Session Memory – Persist user interactions using MemorySaver.

🔍 RAG-Powered Retrieval – Search embedded PDF data and DBs for evidence-backed answers.

🌐 Web Search Integration – Query Tavily for live web results.

⚡ Optimized Context Handling – Handles multi-turn conversations efficiently.

🤖 Groq LLM Integration – High-performance AI responses via Groq.

🛠️ Custom Tool Execution – Execute user-defined or prebuilt tools within LangGraph nodes.

📖 Supported Data Sources

The assistant currently ingests and retrieves knowledge from:

Embedded PDFs:

The Entrepreneurs Guide to Building a Successful Business (2017)

Rich Dad Poor Dad

Pinecone Vector DB for RAG-based retrieval

Live web search via Tavily

This ensures that both curated offline resources and live web knowledge are available for answering questions.


🏗 Architecture

EntrepreneurMentor AI is built with a modular LangGraph pipeline:

User Input
   │
   ▼
[LLM Node] —> Decides whether a tool is needed
   │
   ├─> [Tools Node] → Executes retrievers, web search
   │
   └─> [Summarization Node] → Condenses long conversation
   │
   ▼
AI Response


LLM Node: Handles message processing, context, and optional summarization.

Tools Node: Executes DB or web retrieval tools.

Summarization Node: Reduces conversation length while keeping essential context.

Memory: Persistent storage for session-specific interactions.

⚙️ Installation & Setup

Prerequisites:

Python ≥ 3.11

Streamlit

Access to Groq LLM API, Pinecone API, and Tavily API

# Clone the repo
git clone https://github.com/shoukatkhan/EntrepreneurMentorAI.git
cd EntrepreneurMentorAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt


Environment Variables (.env):

GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
TAVILY_API_KEY=your_tavily_api_key

🖥 Running Locally
streamlit run app.py


Sidebar allows model selection, temperature control, and chat history clearing.

Chat interface supports multi-turn conversations with gradient-styled messages for clarity.

🛠 Usage

Ask startup-related questions in the chat input.

The AI:

Checks if a tool should be used.

Performs DB/web retrieval if required.

Summarizes conversations for long threads.

Returns a concise, evidence-backed answer.

Example Questions:

"What’s the best way to pitch to VCs?"

"How can I improve my product-market fit?"

"Summarize Rich Dad Poor Dad principles for startups."

🧩 Extensibility

EntrepreneurMentor AI is highly modular:

Add custom tools in tools_dict.

Extend retrieval from additional PDF or DB sources.

Swap or fine-tune LLMs for domain-specific tasks.

📈 Real-World Value

Provides practical startup guidance instantly.

Supports founders in decision-making and research.

Can be deployed internally for startup accelerators, incubators, or investor networks.

🔗 Tech Stack
Component	Purpose
Python 3.11+	Core language
Streamlit	Frontend UI
LangGraph 1.0	Agent orchestration
LangChain	LLM integration
Groq LLM	Language model inference
Pinecone	Vector search & retrieval
Tavily Search	Web search integration
dotenv	API key management
📝 Notes

Ensure PDFs are pre-processed into vector embeddings for RAG.

Recommended for startup mentoring and education, not professional legal/financial advice.

Model responses depend on Groq LLM quality and RAG data coverage.

📂 Directory Structure
EntrepreneurMentorAI/
├─ app.py                   # Streamlit main app
├─ prompt.py                # System prompt definition
├─ requirements.txt
├─ .env                     # API keys
├─ src/
│  ├─ nodes/
│  │  ├─ schemas.py
│  │  ├─ decision.py
│  │  └─ retrievers_node.py
├─ data/
│  └─ PDFs/
│     ├─ Entrepreneurs_Guide_2017.pdf
│     └─ Rich_Dad_Poor_Dad.pdf

⚡ Future Improvements

Multi-language support for global founders.

Integration with Slack, Teams, or WhatsApp for team mentoring.

Analytics dashboard to track question trends and AI performance.

Fine-tuned Groq LLM model for startup-specific domain knowledge.

📜 License

MIT License © 2025 Shoukat Khan

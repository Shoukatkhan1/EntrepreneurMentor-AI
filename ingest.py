# 📚 Import all the necessary libraries
from src.loader import load_pdf_file
from src.splitter import text_split
from src.embedding import download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# 🌍 Load secret keys from the .env file
load_dotenv()

# 🗝️ Get the Pinecone API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# 1️⃣ Load and split PDFs
extracted_data = load_pdf_file("Data/")
text_chunks = text_split(extracted_data)

# 2️⃣ Download embeddings
embeddings = download_hugging_face_embeddings()

# 3️⃣ Create a Pinecone index (like a smart search box)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "startupguide-index"  # ✅ all lowercase, valid format

# Check if the index exists; if not, create a new one
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

# 4️⃣ Upload data to Pinecone
# 4️⃣ Upload or load Pinecone vector store
docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

if not os.path.exists("ingestion_done.txt"):
    # Upload documents only if not already uploaded
    docsearch.from_documents(documents=text_chunks)
    with open("ingestion_done.txt", "w") as f:
        f.write("done")
    print("✅ Data ingestion complete and uploaded to Pinecone.")
else:
    print("✅ Data already ingested. Skipping upload.")
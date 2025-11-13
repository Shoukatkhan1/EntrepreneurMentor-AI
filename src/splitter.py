
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

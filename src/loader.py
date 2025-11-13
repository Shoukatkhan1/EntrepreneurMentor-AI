from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


# Extract Data From the PDF File
def load_pdf_file(Data):
    loader = DirectoryLoader(Data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Filter documents to minimal metadata
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        if src is not None:  # Skip documents with no source
            minimal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={"source": src}
                )
            )
    return minimal_docs
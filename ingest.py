# ingest.py
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

user_agent = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
os.environ["USER_AGENT"]

print(f"[DEBUG] User Agent loaded: {user_agent}")

import os, glob, json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredURLLoader

DATA_DIR = "data"
INDEX_DIR = "storage/faiss_index"

def load_urls_from_file(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_documents_from_urls(urls):
    loader = UnstructuredURLLoader(
        urls=urls,
        headers={"User-Agent": user_agent}
    )
    return loader.load()
    # docs = loader.load()
    # for doc in docs:
    #     print(f"Loaded URL: {doc.metadata['source']}")
    #     print(doc.page_content[:300])  # preview first 300 characters
    # return loader.load()

def load_local_docs():
    docs = []
    # PDFs (slides)
    for pdf in glob.glob(os.path.join(DATA_DIR, "slides", "*.pdf")):
        loader = PyPDFLoader(pdf)
        for d in loader.load():
            # keep slide/page metadata
            d.metadata["source"] = os.path.basename(pdf)
            d.metadata["page"] = d.metadata.get("page", None)
            docs.append(d)

    # Word .text files
    for docx in glob.glob(os.path.join(DATA_DIR, "text", "*.docx")):
        try:
            loader = Docx2txtLoader(docx)
            doc_list = loader.load()
            if doc_list is None:
                print(f"[WARN] {docx} returned no docs.")
            else:
                for d in doc_list:
                    d.metadata["source"] = os.path.basename(docx)
                    d.metadata["page"] = None
                    docs.append(d)
        except Exception as e:
            print(f"[WARN] Failed to read {docx}: {e}")
    return docs

def load_all_documents():
    docs = []

    # Existing loaders for PDF, DOCX, TXT, etc.
    docs += load_local_docs()#'data')  # Your current logic

    # Load from URLs
    urls = load_urls_from_file('data/urls.txt')
    if urls:
        print(f"[INFO] Loading {len(urls)} web pages...")
        docs += load_documents_from_urls(urls)

    return docs

def main():
    docs = load_all_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env/secrets
    vs = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"Saved index to {INDEX_DIR}. Chunks: {len(chunks)}")

if __name__ == "__main__":
    main()

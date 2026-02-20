
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

def indexar():
    data_dir = Path("docs")
    docs = []
    for path in data_dir.rglob("*.pdf"):
        print(path)
        loaded_docs = PyPDFLoader(str(path)).load()
        for doc in loaded_docs:
            doc.metadata["filename"] = path.name
        docs.extend(loaded_docs)

    print(f"Documentos {len(docs)} carregados.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print(f"Chunks {len(chunks)} gerados.")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("vectorstore")
    print(f"Indexados {len(vector_store.index_to_docstore_id)} chunks.")


if __name__ == "__main__":
    indexar()
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def db_loader(pdf_path, file_paths, embeddings):
    # Check if the directory is empty
    if not file_paths:
        raise FileNotFoundError(f"No files found in directory: {pdf_path}")

    # Sanity check for each file
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

    ############## Setup Database ##################

    # Configurable paths
    chroma_path = "./chroma_store"
    collection_name = "literature"

    # Load or create vector store
    if os.path.exists(chroma_path):
        try:
            print("📦 Loading existing Chroma vector store...")
            vectorstore = Chroma(
                persist_directory=chroma_path,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            print("✅ Vector store loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load vector store: {str(e)}")
            raise

    else:
        print("🚧 Vector store not found. Creating from PDF documents...")

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        try:
            all_documents = []

            for file_path in file_paths:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                all_documents.extend(pages)

            print(f"📄 Loaded {len(all_documents)} pages from PDFs.")
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            raise

        # Split documents
        pages_split = text_splitter.split_documents(all_documents)

        try:
            os.makedirs(chroma_path, exist_ok=True)

            vectorstore = Chroma.from_documents(
                documents=pages_split,
                embedding=embeddings,
                persist_directory=chroma_path,
                collection_name=collection_name
            )
            print("✅ Created new ChromaDB vector store!")
        except Exception as e:
            print(f"❌ Error creating Chroma vector store: {str(e)}")
            raise

    # Now we create our retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} # K is the amount of chunks to return
    )

    return retriever

def personal_info():
    with open("./docs/peronal_infomation.txt", "r", encoding="utf-8") as my_personal_file:
        return my_personal_file.read()
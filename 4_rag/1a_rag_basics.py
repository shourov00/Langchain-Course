import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # read text content from the file
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # split the text into chunks, chunk size 1000 means 1000 characters per chunk, overlap 0 means no overlap
    text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_spliter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # # Create embeddings (Using openai model)
    # print("\n--- Creating embeddings ---")
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small"
    # )  # Update to a valid embedding model if needed
    # print("\n--- Finished creating embeddings ---")

    # Create embeddings (Using google model)
    print("\n--- Creating embeddings ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. Loading the vector store...")

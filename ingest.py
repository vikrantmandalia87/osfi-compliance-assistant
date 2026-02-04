# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

def ingest_pdf():
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # Create index if doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created index: {index_name}")
    
    # Load PDF
    print("Loading PDF...")
    loader = PyPDFLoader("data/OSFI_Guideline_B-20.pdf")
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} chunks")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # Add this
        openai_api_key=os.getenv("OPENAI_API_KEY")
)
    
    # Get index
    index = pc.Index(index_name)
    
    # Upload in batches
    print("Uploading to Pinecone...")
    batch_size = 100
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]
        
        # Get embeddings
        texts = [doc.page_content for doc in batch]
        embedding_vectors = embeddings.embed_documents(texts)
        
        # Prepare data for upsert
        vectors = []
        for j, (doc, vector) in enumerate(zip(batch, embedding_vectors)):
            vectors.append({
                "id": f"chunk_{i+j}",
                "values": vector,
                "metadata": {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0)
                }
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
    
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest_pdf()
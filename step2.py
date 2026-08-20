import os
from dotenv import load_dotenv
from google import genai
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv('.env')

client = genai.Client()

def get_embedding(text: str) -> list[float]:
    """Generates an embedding vector for a single string using Gemini API."""
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return response.embeddings[0].values

def main():
    # 1. Setup Data: Creating a mock 'article.txt' if you haven't made one yet
    file_path = "article.txt"
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                "Retrieval-Augmented Generation (RAG) is a technique that enhances "
                "large language models (LLMs) by retrieving relevant information from "
                "a knowledge base before generating an answer. This reduces hallucinations.\n\n"
                "ChromaDB is an open-source vector database designed to store and retrieve "
                "vector embeddings efficiently. It runs entirely in-memory for testing, "
                "making it an excellent choice for local RAG development.\n\n"
                "In a RAG system, documents are first chunked into smaller pieces. "
                "Each piece is embedded and stored in a vector database. When a user asks "
                "a question, the system embeds the query and performs a similarity search."
            )
            
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    # 2. Chunking the document
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    # text_splitter.split_text returns a list of strings
    chunks = text_splitter.split_text(document_text)
    print(chunks)
    print(f"Created {len(chunks)} chunks.")

    # 3. Generating embeddings for all chunks
    print("Embedding chunks...")
    chunk_embeddings=[]
    for text in chunks:
        chunk_embeddings.append(get_embedding(text))
    
    # ChromaDB requires a unique string ID for every document chunk
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    print(chunk_ids)   
 
    # 4. Initialize ChromaDB
    print("Initializing ChromaDB...")
    chroma_client = chromadb.Client() # Creates an ephemeral in-memory database
    
    # Clear the collection if it exists so we can re-run the script safely
    try:
        chroma_client.delete_collection("my_documents")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(name="my_documents")

    # 5. Ingestion: Add chunks and their vector embeddings to ChromaDB
    print("Adding chunks to Vector DB...")
    collection.add(
        documents=chunks,
        embeddings=chunk_embeddings,
        ids=chunk_ids
    )

    # 6. Retrieval
    query = "Why is ChromaDB a good choice for local development?"
    print(f"\nUser Query: '{query}'")
    query_embedding = get_embedding(query)

    print("Querying ChromaDB...")
    # Tell Chroma to find the top 1 closest vector to our query embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )
    print(results)

    # ChromaDB returns a dictionary where lists are nested: results["documents"][0][0]
    best_match = None
    if results["documents"] and len(results["documents"][0]) > 0:
        best_match = results["documents"][0][0]
        # Note: Chroma uses L2 distance by default. Lower score = closer match.
        distance = results["distances"][0][0] 
        print(f" - Best Match Distance: {distance:.4f}")
        print(f" - Best Match Text: '{best_match}'")
    else:
        print("No matches found.")

    # 7. Augmented Generation
    prompt = f"""
    Answer the user's question using ONLY the provided Context and Using your Own logical inference.
    If the Context is 'None' or does not contain the answer, reply exactly with: "I do not have enough information to answer that."
    
    Context: {best_match}
    Question: {query}
    """
    
    print("\nGenerating answer with Gemini...")
    generation = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt
    )
    
    print("\n*** FINAL ANSWER ***")
    print(generation.text)

if __name__ == "__main__":
    main()
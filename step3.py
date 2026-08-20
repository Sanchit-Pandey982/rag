import os
from dotenv import load_dotenv
from google import genai
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Import LangSmith's traceable decorator
from langsmith import traceable

load_dotenv('.env')

client = genai.Client()

# 2. Add @traceable to track embedding generation latency and inputs
@traceable(run_type="embedding", name="gemini_embed")
def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embeddings[0].values

# 3. Encapsulate generation and trace it
@traceable(run_type="llm", name="gemini_generate")
def generate_answer(query: str, context: str) -> str:
    prompt = f"""
    Answer the user's question using ONLY the provided Context.
    If the Context is 'None' or does not contain the answer, reply exactly with: "I do not have enough information to answer that."
    
    Context: {context}
    Question: {query}
    """
    
    generation = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return generation.text

def main():
    # --- Ingestion (Same as before) ---
    file_path = "article.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(document_text)
    
    batch_response = client.models.embed_content(model="text-embedding-004", contents=chunks)
    chunk_embeddings = [emb.values for emb in batch_response.embeddings]
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("my_documents")
    except Exception:
        pass
    collection = chroma_client.create_collection(name="my_documents")
    collection.add(documents=chunks, embeddings=chunk_embeddings, ids=chunk_ids)

    # --- 4. The Eval Loop ---
    # We define a list of dictionaries with our test cases.
    eval_dataset = [
        {
            "query": "What is RAG?",
            "expected": "A technique that enhances LLMs by retrieving relevant info to reduce hallucinations."
        },
        {
            "query": "Why use ChromaDB?",
            "expected": "It is an open-source vector database that runs entirely in-memory."
        },
        {
            "query": "How are documents processed in a RAG system?",
            "expected": "They are chunked, embedded, and stored in a vector database."
        },
        {
            "query": "Who is the CEO of Apple?",
            "expected": "I do not have enough information to answer that."
        }
    ]

    print("\nStarting Evaluation Loop...\n")
    print("-" * 50)
    
    # Run the @traceable functions in a loop to log multiple traces in LangSmith
    for i, test_case in enumerate(eval_dataset):
        query = test_case["query"]
        print(f"Test {i+1}: '{query}'")
        
        # Retrieval
        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        best_match = None
        if results["documents"] and len(results["documents"][0]) > 0:
            # Add a distance threshold so unrelated queries drop the context
            distance = results["distances"][0][0]
            if distance < 1.0: # Adjust this threshold based on your Chroma L2 distances
                best_match = results["documents"][0][0]

        # Generation
        actual_answer = generate_answer(query, best_match)
        
        # Results
        print(f"Expected: {test_case['expected']}")
        print(f"Actual:   {actual_answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()
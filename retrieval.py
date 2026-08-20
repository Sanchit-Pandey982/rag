import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv('.env')

# Ensure GEMINI_API_KEY is set in your .env file or environment variables
client = genai.Client()

def get_embedding(text: str) -> list[float]:
    """Generates an embedding vector for a single string using Gemini API."""
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return response.embeddings[0].values

def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates the cosine similarity between two vectors using numpy."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)

def main():
    # 1. The Data (Our local "Database")
    database = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Python is a high-level programming language.",
        "The mitochondria is the powerhouse of the cell."
    ]

    db_embeddings = []
    for text in database:
        db_embeddings.append(get_embedding(text))
    # 2. The User Query
    query = "Who has a world cup messi or ronaldo?"
    
    # Try swapping to this query to test your refusal prompt!
    # query = "Who won the World Cup in 2022?"
    
    print(f"\nUser Query: '{query}'")
    query_embedding = get_embedding(query)

    # 3. The Search Engine Logic
    best_match_idx = -1
    highest_score = -1.0

    print("\nCalculating similarities...")
    for i, db_emb in enumerate(db_embeddings):
        score = calculate_cosine_similarity(query_embedding, db_emb)
        print(f" - Score: {score:.4f} | Sentence: '{database[i]}'")
        
        # BUG FIX: Track the index inside the loop, not the string directly
        if score > highest_score:
            highest_score = score
            best_match_idx = i

    # 4. Threshold logic
    best_match = None
    if highest_score > 0.50 and best_match_idx != -1:
        best_match = database[best_match_idx]

    print(f"\n*** BEST MATCH ***")
    print(f"Sentence: '{best_match}'")
    print(f"Score: {highest_score:.4f}")

    # 5. Prompt Assembly (BUG FIX: Structured explicit prompt)
    prompt = f"""
    Answer the user's question based on the provided Context. You may make reasonable inferences from the Context.
    If the Context is 'None' or the answer cannot be reasonably inferred from the Context, reply exactly with: "I do not have enough information to answer that."  
    
    Context: {best_match}
    Question: {query}
    """
    # 6. Augmented Generation
    print("\nGenerating answer with Gemini...")
    generation = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt
    )
    
    # BUG FIX: .text extracts just the string from the response object
    print("\n*** FINAL ANSWER ***")
    print(generation.text)

if __name__ == "__main__":
    main()
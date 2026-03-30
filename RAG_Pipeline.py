import os           # Lets Python talk to the operating system — we use it to read the API key.
import sys          # Lets us exit the program cleanly if something goes wrong.

import numpy as np                        # NumPy: the foundation of numerical computing in Python.
import faiss                              # FAISS: Facebook AI Similarity Search — production vector search.
from sentence_transformers import SentenceTransformer  # Converts text into semantic embeddings.
from dotenv import load_dotenv            # Reads the .env file and loads secret values like API keys.
from openai import OpenAI                 # The OpenAI-compatible library — works with Groq too.


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:  # If no API key found, stop and tell the user.
    print("ERROR: GROQ_API_KEY not found in .env file.")
    print("Please create a .env file with your key. Copy from .env.example.")
    print("Get your free key at: https://console.groq.com/keys")
    sys.exit(1)  # Exit the program with an error code.

#step 2 - Read the Doc cli
# Pass Filename
#sys.argv 

if len(sys.argv) < 2: #sys.argv[0] [1]
    # len() counts items in a list. If there's only 1 item (the script name itself),
    # the user forgot to provide the filename argument.
    print("Usage: python RAG_Pipeline.py <path_to_text_file>")
    print("Example: python RAG_Pipeline.py Ai_Guide.txt")
    sys.exit(1)  # Exit because we can't continue without a file.

doc_path = sys.argv[1]  # sys.argv[1] is the second item — the filename the user typed.
                         # Example: if user typed "python RAG_Pipeline.py policy.txt"
                         # then sys.argv = ["RAG_Pipeline.py", "policy.txt"]
                         # and sys.argv[1] = "policy.txt"

if not os.path.exists(doc_path):  # Check if the file actually exists on disk.
    print(f"ERROR: File not found: {doc_path}")
    print("Please check the file path and try again.")
    sys.exit(1)  # Exit if the file doesn't exist.

print(f"Loading document: {doc_path}")

with open(doc_path, "r", encoding="utf-8") as f:
    # "r" = read mode. encoding="utf-8" handles all characters including special ones.
    # "with" automatically closes the file when we're done — even if an error occurs.
    full_text = f.read()  # Read the entire file into one big string called full_text.

print(f"Document loaded: {len(full_text)} characters")  # Tell the user how big the document is.
# len(full_text) counts the number of characters in the string.


# Step 3 - SPlit Doc in to Chunks
def split_into_chunks(text, chunk_size=500, overlap=50):
    # This is a FUNCTION. A function is a reusable block of code.
    # "def" means "define a function". "split_into_chunks" is the name.
    # The parentheses hold the inputs (called "parameters"):
    #   text: the full document text string
    #   chunk_size: how many characters each chunk should be (default 500)
    #   overlap: how many characters to share between adjacent chunks (default 50)
    # The function will RETURN a list of chunk strings.

    chunks = []      # Start with an empty list to hold our chunks.
    start = 0        # "start" is the position in the text where the current chunk begins.
                     # We start at position 0 (the very beginning of the text).

    while start < len(text):
        # Keep going as long as "start" hasn't passed the end of the text.
        # len(text) gives the total number of characters.

        end = start + chunk_size  # "end" is where this chunk stops.
                                   # Example: if start=0 and chunk_size=500, end=500.

        chunk = text[start:end]   # text[start:end] is "slicing" — it extracts the substring
                                   # from position start up to (but not including) position end.
                                   # Example: "Hello World"[0:5] = "Hello"

        chunk = chunk.strip()     # .strip() removes extra spaces and newlines from the beginning
                                   # and end of the chunk. This cleans up any leftover whitespace.

        if chunk:                 # "if chunk:" means "if chunk is not empty"
            chunks.append(chunk)  # Add this non-empty chunk to our list.

        start = end - overlap     # Move the start position BACK by "overlap" characters.
                                   # This creates the overlap between chunks.
                                   # Example: if end=500 and overlap=50, next start=450.
                                   # So the next chunk starts at 450, overlapping with this one.

    return chunks  # Return the complete list of chunks back to the caller.
                   # When you call split_into_chunks(text), you get this list.


chunks = split_into_chunks(full_text, chunk_size=500, overlap=50)
# Call our function with the full document text.
# The result (a list of chunk strings) is stored in the variable "chunks".

print(f"Split document into {len(chunks)} chunks")  # Tell the user how many chunks were made.

# step 4 - Create Embeddings FAISS
print("\nLoading embedding model (first run downloads ~30MB model)...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# "all-MiniLM-L6-v2" is a popular, lightweight embedding model.
# It produces 384-dimensional vectors and runs fast even on CPU.
# On first run, it downloads ~30MB. After that, it uses the cached version.
# In production, you might use larger models for better accuracy:
#   - "all-mpnet-base-v2" (420MB, more accurate)
#   - OpenAI's text-embedding-3-small (API-based, no local model needed)

print("Creating embeddings for document chunks...")

chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)
# .encode() converts each chunk text into a 384-dimensional vector.
# show_progress_bar=True shows a progress bar as it processes.
# The result is a NumPy array of shape (num_chunks, 384).
# Each row is one chunk's embedding vector.

# Convert to float32 — FAISS requires this specific data type.
chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)

# Normalize the vectors for cosine similarity search.
# Normalization scales each vector to have length 1.
# After normalization, "inner product" (dot product) = cosine similarity.
faiss.normalize_L2(chunk_embeddings)

# Build the FAISS index.
dimension = chunk_embeddings.shape[1]  # 384 dimensions for all-MiniLM-L6-v2.
# .shape returns (num_chunks, 384). [1] gets the second value = 384.

index = faiss.IndexFlatIP(dimension)
# IndexFlatIP = "Flat Index with Inner Product" search.
# "Flat" means it does an exact search (checks every vector). Best for small datasets.
# "IP" means Inner Product — combined with normalized vectors, this gives cosine similarity.
# For billions of vectors, you'd use IndexIVFFlat or IndexHNSW for approximate but faster search.

index.add(chunk_embeddings)
# .add() stores all the chunk vectors in the FAISS index.
# Now we can search for similar vectors extremely fast.

print(f"FAISS index built with {index.ntotal} vectors ({dimension} dimensions each).")
print("Ready to answer questions!\n")


def search_chunks(query, top_n=3):
    """Search for the most relevant chunks using semantic similarity.

    This function:
    1. Converts the question into an embedding using the same model.
    2. Uses FAISS to find the closest chunk embeddings.
    3. Returns the top_n most relevant chunks with similarity scores.

    Because we use real embeddings (not just keywords):
    - "car" will match chunks about "automobile" or "vehicle"
    - "time off" will match chunks about "vacation" or "leave policy"
    - "salary" will match chunks about "compensation" or "pay"
    """
    # Convert the question into an embedding vector.
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity.

    # Search the FAISS index for the most similar chunk vectors.
    scores, indices = index.search(query_embedding, top_n)
    # .search() returns two arrays:
    #   scores: similarity scores for the top matches (higher = more similar)
    #   indices: the positions (0, 1, 2...) of the matching chunks
    # Both arrays have shape (1, top_n) because we searched one query.

    results = []
    for score, idx in zip(scores[0], indices[0]):
        # scores[0] and indices[0] get results for our single query.
        if idx == -1:  # FAISS returns -1 if there aren't enough results.
            continue
        results.append({
            "chunk_id": f"chunk_{idx}",       # The chunk's ID.
            "chunk_text": chunks[idx],         # The actual text of the chunk.
            "similarity": float(score)         # Cosine similarity score (0 to 1).
        })

    return results

# # ── Step 5: Set up Groq client and system prompt ───────────────────────────

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)
# Create the OpenAI client to communicate with Groq.
# Groq is OpenAI-compatible, so we use the same SDK but point it at Groq's servers.

system_prompt = """You are a document assistant. Your job is to answer questions about a document.

IMPORTANT RULES:
1. Answer questions ONLY based on the provided context from the document.
2. If the answer is not in the provided context, say exactly: "I don't have that information in the document."
3. Do not use any outside knowledge. Only use what is in the context.
4. When you give an answer, briefly mention which part of the document supports your answer.
5. Be concise and helpful.
"""
# This system prompt is critical for RAG. It prevents the AI from "hallucinating" — making up
# information that isn't in the document. By saying "ONLY based on the provided context",
# we force the AI to stick to what the document actually says.


# ── Step 6: Interactive question-answering loop ──────────────────────────────
# Now the program waits for the user to ask questions.
# For each question, we:
#   1. Convert the question into an embedding and search FAISS for similar chunks
#   2. Send those chunks + the question to Groq
#   3. Print Groq's answer
# The user types 'quit' to exit.

print("="*60)
print("  RAG_Pipeline — Document Question Answering System")
print("="*60)
print(f"  Document: {doc_path}")
print(f"  Chunks: {len(chunks)} | Embedding model: all-MiniLM-L6-v2")
print(f"  Vector index: FAISS (IndexFlatIP, {dimension}d)")
print("="*60)
print("Type your question and press Enter. Type 'quit' to exit.")
print("="*60 + "\n")

while True:
    # "while True:" starts an infinite loop. The user keeps asking questions
    # until they type "quit", which triggers "break" to exit the loop.

    question = input("Your question: ")  # Wait for the user to type a question and press Enter.
                                          # Whatever they type is stored in the variable "question".

    question = question.strip()  # Remove any extra spaces or newlines from the input.

    if question.lower() == "quit":
        # .lower() converts the string to lowercase. This means "QUIT", "Quit", "quit" all work.
        # We check if the lowercased question equals the string "quit".
        print("Goodbye! Thanks for using RAG_Pipeline.")
        break  # Exit the while True loop. The program will end.

    if not question:  # If the user just pressed Enter without typing anything...
        print("Please type a question first.")
        continue      # "continue" skips the rest of this loop iteration and goes back to the top.
                       # So we go back to asking for a question without doing anything else.

    # ── Step 6a: Search for relevant chunks ───────────────────────────────
    print("\nSearching document for relevant sections...")

    search_results = search_chunks(question, top_n=3)
    # Call our search function to find the 3 most relevant chunks.
    # This uses FAISS for fast vector similarity search.

    retrieved_chunks = [r["chunk_text"] for r in search_results]
    # List comprehension: extract just the chunk text from the results.
    # This creates a list like ["chunk text 1", "chunk text 2", "chunk text 3"].

    # ── Step 6b: Show the user which chunks were found ────────────────────
    print("\n--- Relevant sections found in document ---")
    for idx, result in enumerate(search_results):
        # Loop through each search result.
        print(f"\n[Source {idx + 1}: {result['chunk_id']}] (similarity: {result['similarity']:.3f})")
        # idx+1 because we want to show 1, 2, 3 not 0, 1, 2.
        # :.3f formats the similarity score to 3 decimal places.
        print(f"{result['chunk_text'][:200]}...")
        # Show just the first 200 characters of the chunk as a preview.
        # [:200] is slicing — characters 0 through 200.
        # "..." indicates the chunk was truncated for display.
    print("--- End of retrieved sections ---\n")

    # ── Step 6c: Build the context string for Groq ──────────────────────
    # We combine all retrieved chunks into one block of text to send to Groq.
    context = "\n\n---\n\n".join(retrieved_chunks)
    # "\n\n---\n\n".join(list) joins the list items with "---" separator lines between them.
    # This makes it visually clear where one chunk ends and the next begins.

    # ── Step 6d: Build the user message for Groq ────────────────────────
    user_message = f"""Here is the relevant context from the document:

CONTEXT:
{context}

QUESTION:
{question}

Please answer the question based only on the context provided above."""
    # We clearly separate the context from the question with labels.
    # This helps Groq understand the structure of the request.

    # ── Step 6e: Send to Groq and get the answer ────────────────────────
    print("Asking Groq...")

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    response = client.chat.completions.create(
        model=model,                  # Which AI model to use.
        max_tokens=1024,              # Maximum length of the answer.
        messages=[
            {
                "role": "system",         # The rules: only answer from context.
                "content": system_prompt
            },
            {
                "role": "user",           # This message is from the user (us).
                "content": user_message   # The context + question we built above.
            }
        ]
    )

    answer = response.choices[0].message.content  # Extract the answer text from the response.

    # ── Step 6f: Print the answer ─────────────────────────────────────────
    print("\n" + "="*60)
    print("ANSWER:")
    print("-"*60)
    print(answer)           # Print Groq's full answer.
    print("="*60)
    print(f"(Tokens used: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output)\n")
    # Show token usage after each question so students can see the cost building up.

# ── Program ends here when the user types 'quit' ────────────────────────────
# The "break" statement in the loop above will cause the while loop to stop,
# and then the program reaches this point and exits naturally.
    
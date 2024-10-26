import os
import openai
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import MarkdownHeaderTextSplitter
import pdfplumber
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys and config from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{PINECONE_INDEX_NAME}' does not exist.")

# Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)

# Extract text from PDF using pdfplumber
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Smarter text chunking using LangChain
def split_into_chunks(text):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(text)
    return chunks


# Store embeddings with metadata in Pinecone
def store_embeddings(chunks):
    for chunk in chunks:
        chunk_text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
        response = openai.embeddings.create(input=chunk, model="text-embedding-ada-002")
        embedding = response.data[0].embedding
        index.upsert([(str(hash(chunk_text)), embedding, {"text": chunk_text})])

# Query Pinecone with keyword arguments and filtering
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = openai.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    return [result['metadata']['text'] for result in results['matches']]


# Generate a long detailed answer using GPT-4
def generate_answer(query):
    chunks = retrieve_relevant_chunks(query)
    context = "\n".join(chunks)
    prompt = f"""


    Query: {query}

    Context from documentation:
    {context}

    Now, generate a well-structured, in-depth answer to the query.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Main logic
if __name__ == "__main__":
    pdf_path = "data/uploaded_pdf.pdf"
    
    print("Extracting text from PDF...")
    pdf_text = extract_pdf_text(pdf_path)

    print("Splitting text into chunks...")
    chunks = split_into_chunks(pdf_text)

    print("Storing embeddings in Pinecone...")
    store_embeddings(chunks)

    print("Ready to answer your questions!")
    query = input("Enter your question: ")
    answer = generate_answer(query)
    print(f"\nAnswer:\n{answer}")

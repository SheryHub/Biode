import os
import uuid
import argparse
from tqdm import tqdm
import torch
import nltk
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def read_text_file(text_path):
    """Extract text from a text file."""
    try:
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error reading text from {text_path}: {e}")
        return ""

def chunk_fixed_length(text, chunk_size=1000, chunk_overlap=100):
    chunks = []
    if len(text) <= chunk_size:
        chunks.append(text)
    else:
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
            else:
                while end > start + chunk_size - 100 and end < len(text) and text[end] != ' ':
                    end -= 1
                if end == start + chunk_size - 100:
                    end = start + chunk_size
                chunks.append(text[start:end])
            start = end - chunk_overlap
    return chunks

def chunk_sentence_based(text, max_chunk_size=1000):
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for sentence in sentences:
        if current_chunk_size + len(sentence) > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_chunk_size = 0
        
        current_chunk.append(sentence)
        current_chunk_size += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def chunk_paragraph_based(text, max_chunk_size=1000):
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for paragraph in paragraphs:
        if current_chunk_size + len(paragraph) > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_chunk_size = 0
        
        current_chunk.append(paragraph)
        current_chunk_size += len(paragraph)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def chunk_semantic_sliding(text, model, max_chunk_size=1000, slide_step=300):
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    window_sentences = []
    window_size = 0
    
    for sentence in sentences:
        window_sentences.append(sentence)
        window_size += len(sentence)
        
        if window_size > max_chunk_size:
            current_chunk = ' '.join(window_sentences)
            chunks.append(current_chunk)
            
            window_sentences = window_sentences[1:]
            window_size = len(' '.join(window_sentences))
    
    if window_sentences:
        chunks.append(' '.join(window_sentences))
    
    return chunks

def chunk_recursive(text, max_chunk_size=1000):
    def recursive_split(input_text, threshold):
        if len(input_text) <= threshold:
            return [input_text]
        
        mid = len(input_text) // 2
        left_half = input_text[:mid]
        right_half = input_text[mid:]
        
        if len(left_half) > threshold:
            left_splits = recursive_split(left_half, threshold)
        else:
            left_splits = [left_half]
        
        if len(right_half) > threshold:
            right_splits = recursive_split(right_half, threshold)
        else:
            right_splits = [right_half]
        
        return left_splits + right_splits
    
    return recursive_split(text, max_chunk_size)

def setup_pinecone_index(api_key, index_name, dimension):
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(name=index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='dotproduct',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Created new Pinecone index: {index_name}")
    else:
        print(f"Using existing Pinecone index: {index_name}")
    index = pc.Index(name=index_name)
    return index

def get_chunking_function(strategy):
    chunking_strategies = {
        'fixed': chunk_fixed_length,
        'sentence': chunk_sentence_based,
        'paragraph': chunk_paragraph_based,
        'recursive': chunk_recursive,
        'semantic_sliding': chunk_semantic_sliding
    }
    return chunking_strategies.get(strategy)

def ingest_text_files(text_dir, index, model, chunking_strategies=['fixed'], namespace=None, chunk_size=1000):
    # Look for .txt files, can be extended to support more text file formats
    text_files = [f for f in os.listdir(text_dir) if f.lower().endswith(('.txt', '.md', '.json', '.csv'))]
    if not text_files:
        print(f"No text files found in {text_dir}")
        return
        
    print(f"Found {len(text_files)} text files to process")
    batch_size = 100
    vectors_for_upsert = []
    
    for text_file in text_files:
        text_path = os.path.join(text_dir, text_file)
        print(f"Processing {text_file}...")
        text = read_text_file(text_path)
        if not text:
            print(f"Skipping {text_file} - no text extracted")
            continue
        
        all_chunks = []
        for strategy in chunking_strategies:
            if strategy == 'semantic_sliding':
                chunks = get_chunking_function(strategy)(text, model, max_chunk_size=chunk_size)
            else:
                chunks = get_chunking_function(strategy)(text, chunk_size)
            
            all_chunks.extend(chunks)
            print(f"Split {text_file} into {len(chunks)} chunks using {strategy} strategy")
        
        for i, chunk in enumerate(tqdm(all_chunks, desc=f"Embedding chunks for {text_file}")):
            chunk_id = f"{text_file.split('.')[0]}_{i}_{uuid.uuid4()}"
            embedding = model.encode(chunk).tolist()
            
            vector_entry = {
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "filename": text_file,
                    "chunk_index": i,
                    "text": chunk[:1000]
                }
            }
            
            vectors_for_upsert.append(vector_entry)
            if len(vectors_for_upsert) >= batch_size:
                index.upsert(vectors=vectors_for_upsert, namespace=namespace)
                vectors_for_upsert = []
    
    if vectors_for_upsert:
        index.upsert(vectors=vectors_for_upsert, namespace=namespace)
    
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats['total_vector_count']}")

def main():
    parser = argparse.ArgumentParser(description="Semantic search on text documents using Pinecone")
    parser.add_argument("--text_dir", type=str, required=True, help="Directory containing text files")
    parser.add_argument("--index_name", type=str, default="kaneki", help="Name for Pinecone index")
    parser.add_argument("--delete_index", action="store_true", help="Delete the index after operation")
    parser.add_argument("--embedding_model", type=str, default="intfloat/e5-small", help="Embedding model to use")
    parser.add_argument("--chunking_strategies", nargs='+', 
                        choices=['fixed', 'sentence', 'paragraph', 'recursive', 'semantic_sliding'], 
                        default=['fixed'], help="Chunking strategies to apply")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Maximum chunk size")
    
    args = parser.parse_args()
    
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Please provide a Pinecone API key via PINECONE_API_KEY environment variable")
        return
    
    # Use CPU by default, can be changed to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SentenceTransformer(args.embedding_model, device=device)
    dimension = model.get_sentence_embedding_dimension()
    print(f"Embedding model: {args.embedding_model}")
    print(f"Embedding dimension: {dimension}")
    
    index = setup_pinecone_index(api_key, args.index_name, dimension)
    
    ingest_text_files(
        args.text_dir, 
        index, 
        model, 
        chunking_strategies=args.chunking_strategies,
        namespace=None,
        chunk_size=args.chunk_size
    )
    
    if args.delete_index:
        pc = Pinecone(api_key=api_key)
        pc.delete_index(name=args.index_name)
        print(f"Deleted index: {args.index_name}")

if __name__ == "__main__":
    main()
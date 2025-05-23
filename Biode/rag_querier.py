import os
import argparse
import torch
import json
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel

class SimpleRAGQuerier:
    def __init__(self, index_name, embedding_model_name, llm_model="gemini-2.0-flash", top_k=5):
        # Initialize Pinecone
        api_key = os.environ.get("PINECONE_API_KEY", "pcsk_5LtThV_5qxFD3giH6hvqpRhpiAptX1U5a73ubXJswGYamduCALfUGvzUi7xNBDhPKZBftT")
        if not api_key:
            raise ValueError("Please set the PINECONE_API_KEY environment variable")
        self.pc = Pinecone(api_key=api_key)
        
        # Connect to existing index
        if not self.pc.has_index(name=index_name):
            raise ValueError(f"Index '{index_name}' not found. Please create it first using the ingestion script.")
        self.index = self.pc.Index(name=index_name)
        
        # Load embedding model
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        print(f"Using device: {device} for embeddings")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        
        # Initialize Vertex AI
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "workshop-458405")
        location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        
        if not project_id:
            raise ValueError("Please set the GOOGLE_CLOUD_PROJECT environment variable")
        
        # Initialize Vertex AI SDK
        aiplatform.init(project=project_id, location=location)
        
        # Set default model parameters
        self.model_name = llm_model
        self.generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 1000,
            "top_k": 40,
            "top_p": 0.95,
        }
        
        self.top_k = top_k
        
        # Keywords that trigger JSON response format
        self.json_keywords = ["coordinate", "coordinates", "location", "position", "gps", "lat", "long", "latitude", "longitude"]
    
    def should_use_json_format(self, question):
        """Check if the question contains keywords that should trigger JSON format response"""
        return any(keyword in question.lower() for keyword in self.json_keywords)
    
    def query(self, question, verbose=False):
        """
        Process a single question through the RAG pipeline.
        """
        # Check if we should use JSON format
        use_json_format = self.should_use_json_format(question)
        
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(question).tolist()
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_metadata=True
        )
        
        # Extract contexts from results
        contexts = []
        for match in results["matches"]:
            if "text" in match["metadata"]:
                contexts.append({
                    "text": match["metadata"]["text"],
                    "score": match["score"],
                    "filename": match["metadata"].get("filename", "Unknown")
                })
        
        if verbose:
            print(f"Retrieved {len(contexts)} relevant chunks")
            for i, ctx in enumerate(contexts):
                print(f"\nContext {i+1} (score: {ctx['score']:.4f}, source: {ctx['filename']}):")
                print(f"{ctx['text'][:200]}...")
        
        if not contexts:
            return "No relevant information found in the database to answer your question."
        
        # Format contexts for the prompt
        formatted_contexts = ""
        for i, ctx in enumerate(contexts, 1):
            formatted_contexts += f"Context {i} (from {ctx['filename']}):\n{ctx['text']}\n\n"
        
        # Create the prompt - adjust based on format needed
        system_prompt = """You are a biodiversity assistant specializing in interpreting species observations, trail-based ecological data, and nature conservation rules. You help users understand where species are typically found, based on trail-specific insights.

                Follow these strict guidelines when generating responses:

                - 🔸 Always respond in **both Urdu and English**, in a clear and informative tone.
                - 🔸 If a user asks **general location questions** (e.g., "Where is it found?" or "Which areas?"), respond using **trail-based descriptions** or **named places** only (e.g., “Margalla Trail 5”, “Daman-e-Koh”).
                - 🔸 **Do NOT include coordinates** or raw location data (like lat/lon) unless the user **explicitly asks** for “coordinates”, “specific location”, “latitude”, “longitude”, “map”, or “مخصوص مقام”.
                - 🔸 If the context lacks enough relevant data to answer, respond with:  
                  **"I don't have enough information to answer this question."**  
                  **"مجھے اس سوال کا جواب دینے کے لیے کافی معلومات دستیاب نہیں ہیں۔"**
                - 🔸 Do **not fabricate** or infer information that isn't directly available in the context.
                - 🔸 Prioritize human-readable summaries and nature trail connections over data points.
                - 🔸 Avoid repeating both coordinate and trail info—choose **only one**, based on the user's question.

                Be concise, helpful, and accurate in every answer.
                """


        if self.should_use_json_format(question):
            system_prompt += """

For this specific question, you must respond in valid JSON format with the following structure:
{
  "species": [
    {
      "name": "Scientific or common name of species",
      "location": {
        "latitude": latitude_value_as_float,
        "longitude": longitude_value_as_float
      }
    },
    ... additional species if available ...
  ]
}

If exact coordinates are not available in the context but a general location is mentioned, you can use approximate coordinates and note this in a separate "note" field in each species entry. If no location data is available, set latitude and longitude to null but include any species mentioned."""
        
        user_prompt = f"""Based on the following contexts, please answer this question: {question}

{formatted_contexts}

Answer:"""
        
        # Call Vertex AI model
        model = GenerativeModel(self.model_name)
        
        # Format prompt for Vertex AI
        prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=self.generation_config
        )
        
        answer = response.text
        
        # Process JSON response if needed
        if self.should_use_json_format(question):
            try:
                # Try to extract JSON from the response if it's wrapped in markdown code blocks
                if "```json" in answer:
                    json_text = re.search(r'```json\n(.*?)\n```', answer, re.DOTALL)
                    if json_text:
                        answer = json_text.group(1)
                
                # Validate JSON
                json_obj = json.loads(answer)
                answer = json.dumps(json_obj, indent=2)
                
                # For JSON responses, we'll return without source information
                return answer
            except json.JSONDecodeError:
                # If JSON parsing fails, try to fix the response
                model = GenerativeModel(self.model_name)
                fix_prompt = f"""The following JSON is invalid. Please fix it to make it valid according to this structure:
{{
  "species": [
    {{
      "name": "Scientific or common name of species",
      "location": {{
        "latitude": latitude_value_as_float,
        "longitude": longitude_value_as_float
      }}
    }}
  ]
}}

Invalid JSON:
{answer}

Valid JSON:"""
                
                fix_response = model.generate_content(fix_prompt)
                try:
                    answer = fix_response.text
                    if "```json" in answer:
                        json_text = re.search(r'```json\n(.*?)\n```', answer, re.DOTALL)
                        if json_text:
                            answer = json_text.group(1)
                    
                    json_obj = json.loads(answer)
                    answer = json.dumps(json_obj, indent=2)
                    return answer
                except:
                    # If still failing, return a structured error
                    error_json = {
                        "error": "Could not extract valid location data",
                        "message": "The system could not find precise coordinate information for your query."
                    }
                    return json.dumps(error_json, indent=2)
        
        return f"{answer}"

def main():
    parser = argparse.ArgumentParser(description="Simple RAG querier for Pinecone indexes")
    parser.add_argument("--index_name", type=str, required=True, help="Name of your Pinecone index")
    parser.add_argument("--embedding_model", type=str, default="intfloat/e5-large", help="Embedding model name")
    parser.add_argument("--llm_model", type=str, default="gemini-pro", help="Vertex AI model name (e.g., gemini-pro, text-bison)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of contexts to retrieve")
    parser.add_argument("--query", type=str, help="Single query to run (optional)")
    parser.add_argument("--verbose", action="store_true", help="Print verbose information about retrieved chunks")
    
    args = parser.parse_args()
    
    querier = SimpleRAGQuerier(
        index_name=args.index_name,
        embedding_model_name=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k
    )
    
    if args.query:
        # Run a single query
        answer = querier.query(args.query, verbose=args.verbose)
        print("\nAnswer:")
        print(answer)
    else:
        # Interactive mode
        print("\nRAG Q&A System")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nEnter your question: ")
            if question.lower() in ['exit', 'quit']:
                break
                
            answer = querier.query(question, verbose=args.verbose)
            print("\nAnswer:")
            print(answer)

if __name__ == "__main__":
    main()

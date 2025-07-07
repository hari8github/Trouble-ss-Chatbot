import chromadb
import json
from chromadb.utils import embedding_functions
import requests
from typing import List, Union
import re

class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name="llama2"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/embeddings"
        
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        categories = {
            'hardware': {
                'overheating': ['overheat', 'hot', 'temperature', 'fan noise', 'cooling'],
                'display': ['screen', 'display', 'monitor', 'blank', 'flickering'],
                'power': ['battery', 'charging', 'power', 'shutdown', 'drain']
            },
            'software': {
                'bsod': ['blue screen', 'bsod', 'crash', 'system error'],
                'performance': ['slow', 'freeze', 'lag', 'not responding', 'hanging']
            },
            'network': ['wifi', 'internet', 'connection', 'ethernet', 'network'],
            'ports': {
                'usb': ['usb port', 'usb not working', 'device not recognized'],
                'audio': ['headphone', 'speaker', 'sound', 'audio jack'],
                'display_port': ['hdmi', 'display port', 'monitor connection']
            },
            'peripherals': {
                'mouse': ['mouse not working', 'cursor', 'click', 'scroll'],
                'keyboard': ['keyboard', 'keys', 'typing', 'keypress'],
                'printer': ['printer', 'printing', 'scanner']
            }
        }
        
        for category, subcats in categories.items():
            if isinstance(subcats, dict):
                for subcat, keywords in subcats.items():
                    if any(kw in text for kw in keywords):
                        return f"{category}_{subcat}_issue: {text}"
            elif isinstance(subcats, list):
                if any(kw in text for kw in subcats):
                    return f"{category}_issue: {text}"
        return text
        
    def enhance_prompt(self, text: str) -> str:
        return f"""
        Task: Generate comprehensive troubleshooting embeddings
        Context: General technical problem diagnosis
        Input: {text}
        """
    
    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        
        embeddings = []
        for text in input:
            try:
                processed_text = self.preprocess_text(text)
                enhanced_prompt = self.enhance_prompt(processed_text)
                
                response = requests.post(
                    self.url,
                    json={"model": self.model_name, "prompt": enhanced_prompt},
                    timeout=15
                )
                
                if response.status_code == 200:
                    embedding = response.json()["embedding"]
                    embeddings.append(embedding)
                else:
                    print(f"Error from API: {response.text}")
                    raise Exception(f"API Error: {response.status_code}")
                    
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                raise
        
        return embeddings

def process_query(query_text: str) -> dict:
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=3
        )
        return results
    except Exception as e:
        print(f"Query error: {str(e)}")
        return None

def format_results(results: dict) -> str:
    if not results or 'documents' not in results or not results['documents'][0]:
        return "\nNo matching solutions found. Please try rephrasing your query."
    
    output = "\n=== Troubleshooting Results ===\n"
    for idx, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        output += f"\n{idx + 1}. Issue: {meta['problem']}\n"
        output += f"Category: {meta['category']}\n"
        output += f"Solution: {doc}\n"
        output += "-" * 50 + "\n"
    return output


client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = OllamaEmbeddingFunction(model_name="llama2")

with open("./troubleshooting.json", "r", encoding="utf-8") as f:
    solutions_data = json.load(f)


try:
    collection = client.get_collection("troubleshooting")
    print("Using existing ChromaDB collection.")
except Exception:
    print("Collection not found. Creating a new one.")
    collection = client.create_collection(
        name="troubleshooting",
        embedding_function=embedding_function
    )

documents, ids, metadata = [], [], []
for i, category in enumerate(solutions_data["solutions"]):
    for j, problem in enumerate(category["problems"]):
        text = (
            f"Issue: {problem['problem']}\n"
            f"Category: {category['category']}\n"
            f"Solution: {' '.join(problem['steps'])}\n"
        )
        documents.append(text)
        ids.append(f"doc_{i}_{j}")
        metadata.append({"category": category["category"], "problem": problem["problem"]})

for i in range(0, len(documents), 10):
    collection.add(
        documents=documents[i:i+10],
        ids=ids[i:i+10],
        metadatas=metadata[i:i+10]
    )
    print(f"Added {len(documents[i:i+10])} documents to collection.")

if __name__ == "__main__":
    print(f"Total documents in collection: {collection.count()}")
    print("\nTroubleshooting Bot Ready!")
    print("Enter your problem (or 'exit' to quit):")
    
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == 'exit':
            break
            
        results = process_query(query)
        print("Debug - Results:", results) 
        print(format_results(results))
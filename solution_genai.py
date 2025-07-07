import requests
import json
import chromadb
from vector_store import OllamaEmbeddingFunction
import re


embedding_function = OllamaEmbeddingFunction()

def preprocess_text(text: str) -> str:
    return embedding_function.preprocess_text(text).strip().lower() 

try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("troubleshooting")
except Exception as e:
    print(f"Error loading ChromaDB collection: {e}")
    collection = None

TICKET_PORTAL_URL = "https://support.example.com/raise-ticket" 


def retrieve_solution(issue_text: str):
    """Queries ChromaDB to retrieve structured solutions for the issue."""
    if collection is None:
        print("Warning: ChromaDB collection not available.")
        return None
    
    try:
        print(f"[DEBUG] Querying ChromaDB with: {issue_text}")  
        results = collection.query(query_texts=[issue_text], n_results=3, include=["documents"])
        print(f"[DEBUG] Raw ChromaDB Results: {results}")  

        if results and results.get("documents"):
            for doc_list in results["documents"]:
                for doc in doc_list:  
                    print(f"[DEBUG] Processing Document: {doc}")  
                    
                    if isinstance(doc, str):
        
                        if "Solution:" in doc:
                            solution_text = doc.split("Solution:", 1)[-1].strip()
                            steps = [step.strip() for step in solution_text.split(". ") if step]
                            print(f"[DEBUG] Extracted Steps: {steps}") 
                            return {"solution_steps": steps}
                        
                     
                        try:
                            json_data = json.loads(doc)
                            print(f"[DEBUG] Extracted JSON: {json_data}")  
                            return json_data
                        except json.JSONDecodeError:
                            print(f"[DEBUG] Failed to parse JSON from document: {doc}")
                            continue
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
    
    return None


def is_issue_resolved(user_response: str) -> bool:
    """Checks if the user's response indicates the issue is resolved."""
    positive_patterns = [
        r"\b(fixed|solved|working|resolved|thank you|it's fine|ok now|done)\b"
    ]
    
    negative_patterns = [
        r"\b(not working|still an issue|didn't help|same problem|no change|not fixed)\b"
    ]
    
    response = user_response.lower().strip()
    
    if any(re.search(pattern, response) for pattern in positive_patterns):
        return True 
    elif any(re.search(pattern, response) for pattern in negative_patterns):
        return False  
    else:
        return None 

def provide_solution(user_issue: str, say, thread_ts):
    """Handles the solution retrieval and presents one step at a time in Slack."""
    issue_text = preprocess_text(user_issue)
    solution_data = retrieve_solution(issue_text)
    
    if not solution_data:
        say(f"Bot: I couldn't find a matching solution in the database. Please check the issue description or raise a support ticket: {TICKET_PORTAL_URL}", thread_ts=thread_ts)
        return
    
    steps = solution_data.get("solution_steps", [])
    if not steps:
        say("Bot: No detailed steps available for this issue. You may need to raise a support ticket.", thread_ts=thread_ts)
        return

    say("Bot: Let's go through the solution step by step.", thread_ts=thread_ts)

  
    seen_steps = set()

    for step in steps:
        if step in seen_steps:
            continue  

        say(f"Bot: {step}", thread_ts=thread_ts)
        seen_steps.add(step)

      
        user_response = input("User: ").strip().lower()
        ack = is_issue_resolved(user_response)

        if ack is True:
            say("Bot: Glad I could help! Let me know if you need anything else.", thread_ts=thread_ts)
            return
        elif ack is False:
            continue  

    say(f"Bot: Since the issue persists, you can raise a support ticket here: {TICKET_PORTAL_URL}", thread_ts=thread_ts)


if __name__ == "__main__":
    user_issue = input("Describe your issue: ")
    provide_solution(user_issue)

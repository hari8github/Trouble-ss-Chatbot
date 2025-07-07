import requests
import json
import base64
import chromadb
from vector_store import OllamaEmbeddingFunction
import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


embedding_function = OllamaEmbeddingFunction()


def preprocess_text(text: str) -> str:
    return embedding_function.preprocess_text(text)

try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("troubleshooting")
except Exception as e:
    print(f"Error loading ChromaDB collection: {e}")
    collection = None

class QuestionManager:
    def __init__(self):
        self.qa_history = []
        self.asked_questions = set()
        self.summary = ""

    def add_qa(self, question: str, answer: str):
        if question and answer:
            self.qa_history.append({"question": question, "answer": answer})
            self.asked_questions.add(question)

    def format_qa_history(self) -> List[Dict[str, str]]:
        """Returns the raw Q&A history list."""
        return self.qa_history

VISION_API_URL = "http://localhost:11434/api/generate"
LLM_API_URL = "http://localhost:11434/api/generate"


def extract_text_from_image(image_path: str) -> str:
    """Sends an image to the vision model and extracts text."""
    try:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")  
        
        print("Sending image to Ollama LLaVA model...")
        response = requests.post(VISION_API_URL, json={
            "model": "llava",
            "prompt": "Extract only the text from this image and nothing more than that.:",
            "images": [image_data]  
        }, timeout=60)  # Increased timeout to 60 seconds

        print(f"API Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                responses = response.text.strip().split("\n")  
                extracted_text = ""

                for resp in responses:
                    try:
                        data = json.loads(resp) 
                        extracted_text += data.get("response", "") + " " 
                    except json.JSONDecodeError:
                        continue  

                extracted_text = extracted_text.strip()
                print(f"Extracted Text: {extracted_text}")  
                return extracted_text
            except Exception as e:
                print(f"Error processing API response: {e}")
                return "Error processing the image content."
        else:
            print(f"Error from Ollama API: {response.text}")
            return "Image processing failed. Ollama API returned an error."
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error to Ollama: {e}")
        return "Unable to connect to the vision model. Please ensure Ollama is running."
    except requests.exceptions.Timeout:
        print("Connection to Ollama timed out")
        return "Image processing timed out. The model may be busy or the image is too complex."
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return "Image processing failed due to an unexpected error."

def generate_question(context: str, previous_qa: list, question_stage: int) -> str:
    """Generates staged troubleshooting questions with retry logic."""
    max_retries = 3
    retry_delay = 2  

    prompt = f"""
    You are a helpful IT support assistant. Generate ONE simple question for stage {question_stage}/3.
    
    Current Issue: {context}
    Previous Q&A: {previous_qa}

    Rules:
    1. Questions must be:
       - Simple and easy to answer (yes/no or short response)
       - Focus on one thing at a time
       - User-friendly, not technical
    
    2. Question progression:
       Stage 1: Basic verification (e.g., "Have you tried restarting your device?")
       Stage 2: Issue scope (e.g., "Does this happen on other websites too?")
       Stage 3: Timeline (e.g., "Did this start today or has it been happening for a while?")

    Example good questions:
    - "Have you tried turning off your WiFi and turning it back on?"
    - "Can other people in your location access the internet?"
    - "Does this happen every time you try to connect?"

    Example bad questions:
    - "What specific error codes are you seeing?" (too technical)
    - "Can you describe all the troubleshooting steps you've taken?" (too broad)
    - "What happens when you run ipconfig in command prompt?" (too technical)

    Return ONLY the question, no prefixes or additional text.
    """

    for attempt in range(max_retries):
        try:
            response = requests.post(
                LLM_API_URL,
                json={"model": "llama3", "prompt": prompt, "stream": False},
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, dict) and "response" in data:
         
                question = data["response"].strip()
          
                prefixes = ["Bot:", "Question:", "Stage", "Follow-up:"]
                for prefix in prefixes:
                    if question.startswith(prefix):
                        question = question[len(prefix):].strip()
                return question.strip()
            
            logger.warning(f"Unexpected API response format on attempt {attempt + 1}")
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            
        except Exception as e:
            logger.error(f"Error generating question on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    
    return "I'm having trouble connecting to my system. Please try again."

def ask_next_question(thread_ts, say, context): 
    """Generates the next troubleshooting question."""

    if "question_manager" not in context:
        context["question_manager"] = QuestionManager()
    

    next_question = generate_question(
        context=context["user_issue"],
        previous_qa=context["question_manager"].format_qa_history(),
        question_stage=context["question_manager"].stage
    )
    

    say(text=next_question, thread_ts=thread_ts)
    
  
    context["question_manager"].stage += 1


def retrieve_past_interactions(issue_text: str):
    """Queries ChromaDB to retrieve past troubleshooting interactions related to the issue."""
    if collection is None:
        print("Warning: ChromaDB collection not available.")
        return ""
    
    try:
        results = collection.query(query_texts=[issue_text], n_results=3, include=["documents", "metadatas"])
        if results and "documents" in results and results["documents"] and results["documents"][0]:
            return "\n".join(results["documents"][0])
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
    
    return ""


def start_questioning(user_input: str, image_path: str = None):
    """Starts the questioning process based on user input (text or image)."""
    question_manager = QuestionManager()
  
    if image_path:
        extracted_text = extract_text_from_image(image_path)
        if extracted_text == "Image processing failed." or not extracted_text:
            print("Bot: I couldn't process the image. Please try another one.")
            return
        issue_text = preprocess_text(extracted_text)
    else:
        issue_text = preprocess_text(user_input)
    
    past_interactions = retrieve_past_interactions(issue_text)
    
 
    while question_manager.stage <= 3:
    
        question = generate_question(
            context=issue_text,
            previous_qa=question_manager.format_qa_history(),
            question_stage=question_manager.stage
        )
        
        print(f"Bot: {question}")
        user_response = input("User: ")
        
        question_manager.add_qa(question, user_response)
        question_manager.stage += 1
    
    print("Bot: Thank you for providing the information. I'll analyze this now.")
    return question_manager.qa_history

if __name__ == "__main__":
    user_input = input("Describe your issue (or type 'image' to upload a photo): ")
    
    if user_input.lower() == "image":
        image_path = input("Enter the path of the image file: ")
        start_questioning(user_input="", image_path=image_path)
    else:
        start_questioning(user_input)
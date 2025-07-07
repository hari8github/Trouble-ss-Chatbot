from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
import time
import tempfile
from typing import Dict, List, Optional
from dotenv import load_dotenv
from question_genai import generate_question, extract_text_from_image, start_questioning, QuestionManager
from solution_genai import provide_solution, retrieve_solution
import logging
import json
import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")


app = App(token=SLACK_BOT_TOKEN)
conversation_states: Dict[str, Dict] = {}


def start_troubleshooting(user_issue: str, say, thread_ts: str) -> None:
    """Initializes a new troubleshooting conversation."""
    logger.info(f"🎯 Starting new troubleshooting session for issue: {user_issue}")
    

    clean_issue = user_issue.replace("The image contains the following text:", "").replace("This appears to be", "").strip()
    

    conversation_states[thread_ts] = {
        "user_issue": clean_issue,
        "thread_ts": thread_ts,
        "questioning": True,
        "question_count": 0,
        "solution_steps": [],
        "current_step": 0,
        "resolved": False,
        "question_manager": QuestionManager()
    }
    
 
    context = conversation_states[thread_ts]
    
   
    ask_next_question(thread_ts, say, context)

def ask_next_question(thread_ts: str, say, context: Dict) -> None:
    """Generates and asks the next question based on context."""
    try:
 
        context["question_count"] += 1
        

        logger.info(f"Generating question {context['question_count']} for issue: {context['user_issue']}")
        

        previous_qa = context["question_manager"].format_qa_history()
        
   
        next_question = generate_question(
            context=context["user_issue"],
            previous_qa=previous_qa,
            question_stage=context["question_count"]
        )
        
        logger.info(f"Generated question: {next_question}")
        
        if next_question:
            context["current_question"] = next_question
            say(next_question, thread_ts=thread_ts)
        else:
            logger.warning("No question generated, moving to solutions")
            provide_solution(context["user_issue"], say, thread_ts)
    except Exception as e:
        logger.error(f"❌ Error generating question: {str(e)}", exc_info=True)
        say("I'm having trouble generating questions. Let's proceed with solutions.", thread_ts=thread_ts)
        provide_solution(context["user_issue"], say, thread_ts)

def validate_conversation_state(context: Dict) -> bool:
    """Validates that all required keys are present in the conversation state."""
    required_keys = [
        "user_issue", "thread_ts", "questioning", 
        "question_count", "solution_steps", "current_step",
        "resolved", "question_manager"
    ]
    return all(key in context for key in required_keys)

def cleanup_old_conversations():
    """Removes conversations older than 1 hour."""
    current_time = time.time()
    for thread_ts in list(conversation_states.keys()):
        if current_time - float(thread_ts) > 3600:  
            del conversation_states[thread_ts]

def handle_image_upload(file: dict, say, thread_ts: str) -> None:
    """Handles image uploads and extracts text from them."""
    try:
        image_url = file.get("url_private")
        if not image_url:
            logger.error("❌ No image URL found in file object")
            say("Sorry, I couldn't process the image. Please try again.", thread_ts=thread_ts)
            return

        logger.info(f"📸 Processing image from URL: {image_url}")
        
 
        extracted_text = extract_text_from_slack_image(image_url)
        if not extracted_text or extracted_text == "Image processing failed.":
            say("Sorry, I couldn't extract any text from the image. Please try typing your issue instead.", 
                thread_ts=thread_ts)
            return

 
        logger.info(f"Starting troubleshooting with extracted text: {extracted_text}")
        start_troubleshooting(extracted_text, say, thread_ts)

    except Exception as e:
        logger.error(f"❌ Error processing image upload: {str(e)}", exc_info=True)
        say("Sorry, I couldn't process the image. Please try again or type your issue instead.", 
            thread_ts=thread_ts)

@app.event("message")
def handle_all_messages(event, say):
    """Handles incoming Slack messages and images."""
    try:
        logger.info(f"💬 Message received: {json.dumps(event, indent=4)}")
        thread_ts = event.get("thread_ts") or event.get("ts")

  
        cleanup_old_conversations()

 
        if "files" in event:
            for file in event["files"]:
                if file.get("mimetype", "").startswith("image/"):
                    handle_image_upload(file, say, thread_ts)
                    return

        user_message = event.get("text", "").strip()
        if user_message:
            logger.info(f"💬 User message detected: {user_message}")
            context = conversation_states.get(thread_ts)
            if not context:
                start_troubleshooting(user_message, say, thread_ts)
            else:
                handle_user_response(user_message, thread_ts, say)
    except Exception as e:
        logger.error(f"Error in message handler: {str(e)}")
        say("Sorry, something went wrong. Please try again.", thread_ts=thread_ts)


def is_solution_successful(response: str) -> bool:
    """Check if user's response indicates the solution worked."""
    success_indicators = [
        "worked", "fixed", "solved", "resolved", "yes", "done", 
        "great", "perfect", "thank", "good"
    ]
    response = response.lower()
    return any(indicator in response for indicator in success_indicators)


def handle_user_response(user_response: str, thread_ts: str, say):
    """Processes user responses and manages conversation flow."""
    try:
        context = conversation_states.get(thread_ts)
        if not context or not validate_conversation_state(context):
            logger.warning("⚠️ Invalid conversation state")
            return

        context["last_response"] = user_response

        if context.get("waiting_for_ack", False):
            context["waiting_for_ack"] = False
            

            if is_solution_successful(user_response):
                say("Great! 🎉 I'll close this ticket now. If you need help again, feel free to start a new conversation.", 
                    thread_ts=thread_ts)
                context["resolved"] = True
                return
            

            context["current_step"] += 1
            provide_solution(context["user_issue"], say, thread_ts)
        elif context["question_count"] < 3:
            context["question_manager"].add_qa(context.get("current_question"), user_response)
            ask_next_question(thread_ts, say, context)
        else:
            provide_solution(context["user_issue"], say, thread_ts)
    except Exception as e:
        logger.error(f"Error handling user response: {str(e)}")
        say("Sorry, something went wrong. Please try again.", thread_ts=thread_ts)

def provide_solution(user_issue: str, say, thread_ts: str):
    """Provides solutions in grouped steps (2-2-1 format), waiting for acknowledgment."""
    context = conversation_states.get(thread_ts)
    if not context:
        logger.warning("⚠️ No conversation context found")
        return

    issue_text = user_issue.strip().lower()
    solution_data = retrieve_solution(issue_text)
    logger.info(f"🔍 Retrieved solution data: {solution_data}")

    steps = solution_data.get("solution_steps", []) if solution_data else []
    if not steps:
        summary = generate_conversation_summary(context)
        say(f"""
*📝 Troubleshooting Summary:*
{summary}
{get_support_info(is_final=False)}
        """, thread_ts=thread_ts)
        return


    step_groups = [
        (0, 2),  
        (2, 4), 
        (4, 5)   
    ]

    if context["current_step"] < len(step_groups):
        start, end = step_groups[context["current_step"]]
        current_steps = steps[start:end]
        
        if "solution_steps" not in context:
            context["solution_steps"] = []
        
        
        for step in current_steps:
            if step not in context["solution_steps"]:
                context["solution_steps"].append(step)
        

        steps_text = "\n".join([f"• {step}" for step in current_steps])
        
        say(f"""
*🔧 Solution {context['current_step'] + 1}:*
{steps_text}

*Please try these steps and respond with any message to continue.*
        """, thread_ts=thread_ts)
        context["waiting_for_ack"] = True
    else:
        summary = generate_conversation_summary(context)
        say(f"""
*📝 Final Troubleshooting Summary*
{summary}

*Need Additional Help?*
📌 *Create Support Ticket:* https://elluciansupport.service-now.com/esc
📌 *Join Support Channel:* #help-it (/ticket)
📌 *Contact Support Team:* help-it@ellucian.com
        """, thread_ts=thread_ts)
        context["resolved"] = True

def generate_conversation_summary(context: Dict) -> str:
    """Generates a concise summary of the troubleshooting conversation."""
    if not context or not isinstance(context, dict):
        return "No context available"
        

    issue = context.get('user_issue', 'No issue specified')
    clean_issue = issue.split('.')[0].strip()
    clean_issue = clean_issue.replace("The text in the image is:", "").strip()
    clean_issue = clean_issue.replace("\"", "").strip()
    

    qa_history = context.get("question_manager").format_qa_history()
    findings = _format_qa_findings(qa_history)
    
    solutions = _format_solutions(context.get('solution_steps', []))
    
    summary = [
        f"🔹 *Issue:* {clean_issue}",
        f"🔹 *Findings:* {findings}",
        f"🔹 *Solutions Tried:*{solutions}"
    ]
    
    return "\n".join(summary)

def _format_qa_findings(qa_history: List) -> str:
    """Formats Q&A history into a concise findings statement using LLaMA3."""
    if not qa_history or not isinstance(qa_history, list):
        logger.info("No QA history found")
        return "Insufficient troubleshooting information"
    
    try:

        qa_text = ""
        for qa in qa_history:
            if isinstance(qa, dict):
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "").strip()
                if question and answer:
                    qa_text += f"Question: {question}\nAnswer: {answer}\n\n"
        
        if not qa_text:
            logger.info("No valid Q&A pairs found")
            return "No diagnostic information available"


        prompt = f"""Based on this technical support conversation, create a single sentence that summarizes the key findings:

{qa_text}

Requirements:
- Focus only on facts from user's answers
- Describe what was discovered about the issue
- Include timing, scope, and impact if mentioned
- Make it technical but clear

Example good summary: "System crashes began after recent updates and occur randomly across all programs"

Summary:"""

        logger.info("Sending request to LLaMA3...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.4,
                "max_tokens": 300
            },
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"LLaMA3 API error: {response.status_code}")
            return "Error generating findings summary"

        data = response.json()
        summary = data.get("response", "").strip()

        prefixes = ["Summary:", "Finding:", "The issue"]
        for prefix in prefixes:
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix):].strip()

        logger.info(f"Generated summary: {summary}")
        return summary if summary else "Could not generate findings summary"

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to LLaMA3 API. Is Ollama running?")
        return "System diagnostic summary unavailable - could not connect to local AI service"
    except Exception as e:
        logger.error(f"Error in _format_qa_findings: {str(e)}", exc_info=True)
        return "Error generating findings summary"
        
def _format_solutions(solutions: List) -> str:
    """Formats attempted solutions into a bulleted list."""
    if not solutions:
        return "No solutions attempted"
    return "\n• " + "\n• ".join(solutions)

def extract_text_from_slack_image(image_url: str) -> str:
    """Fetch the image from Slack and extract text."""
    headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
    temp_file = None

    try:
        response = requests.get(image_url, headers=headers, stream=True)
        if response.status_code != 200:
            logger.error(f"❌ Failed to download image. Status Code: {response.status_code}")
            return "Image processing failed."

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        for chunk in response.iter_content(1024):
            temp_file.write(chunk)
        temp_file.close()

        extracted_text = extract_text_from_image(temp_file.name)
        logger.info(f"📝 Extracted text: {extracted_text}")
        return extracted_text
    except Exception as e:
        logger.error(f"❌ Error processing image: {str(e)}")
        return "Image processing failed."
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    logger.info("🔹 Starting Slack bot...")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
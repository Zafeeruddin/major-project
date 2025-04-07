from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
from datetime import datetime
import sys
import os
import dateparser
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from final_model_generation import predict

# Disable TorchVision dependency
transformers.image_utils._is_torch_available = False

# Configuration
DATETIME_FORMAT = "%d-%m-%Y %H:%M"
DEFAULT_TIME = "12:00"  # Use noon as default time if not specified

# Enhanced greeting detection
GREETINGS = {
    "hello", "hi", "hey", "greetings", "good morning", 
    "good afternoon", "good evening", "howdy", "hola"
}

# Improved footfall-related keywords
FOOTFALL_KEYWORDS = {
    "footfall", "visitors", "crowd", "people count", 
    "attendance", "patrons", "customer flow"
}

def predict_footfall(datetime_str):
    """Get footfall prediction with proper error handling"""
    try:
        footfall = predict.predict_ensemble_for_datetime(datetime_str)
        return f"The predicted footfall for {datetime_str} is {footfall}."
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Sorry, I'm having trouble accessing footfall predictions right now. Please try again later."

def classify_prompt(prompt):
    """Classify user prompt with improved detection"""
    prompt = prompt.lower().strip()
    
    # Check for greetings
    if any(re.search(rf'\b{greeting}\b', prompt) for greeting in GREETINGS):
        return "greeting"
    
    # Check for footfall-related queries
    if any(keyword in prompt for keyword in FOOTFALL_KEYWORDS):
        return "footfall_count"
    
    return "other"

def extract_datetime(prompt):
    """Extract and format datetime using advanced parsing"""
    try:
        # Normalize prompt for better parsing
        prompt = re.sub(r'\b(?:for|at|on|about)\b', '', prompt, flags=re.IGNORECASE)
        
        # Parse with dateparser
        settings = {
            'RELATIVE_BASE': datetime.now(),
            'PREFER_DATES_FROM': 'future',
            'PREFER_DAY_OF_MONTH': 'current'
        }
        parsed_date = dateparser.parse(prompt, settings=settings)
        
        if not parsed_date:
            return None
            
        # Add default time if none detected
        if parsed_date.time() == datetime.min.time():
            default_time = datetime.strptime(DEFAULT_TIME, "%H:%M").time()
            parsed_date = parsed_date.replace(
                hour=default_time.hour,
                minute=default_time.minute
            )
            
        return parsed_date.strftime(DATETIME_FORMAT)
    except Exception as e:
        print(f"Datetime extraction error: {str(e)}")
        return None

def format_response(response):
    """Format model response for better readability"""
    return response.split("<|im_end|>")[0].replace("<|im_start|>", "").strip()

# Load model and tokenizer
model_name = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
).eval()

# Main interaction loop
def main():
    print("Welcome to the Footfall Prediction Assistant! Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in {'exit', 'quit', 'bye'}:
                print("Goodbye!")
                break

            # Classify and handle input
            prompt_type = classify_prompt(user_input)
            
            if prompt_type == "greeting":
                print("Bot: Hello! How can I assist you with footfall predictions today?")
                
            elif prompt_type == "footfall_count":
                datetime_str = extract_datetime(user_input)
                
                if datetime_str:
                    response = predict_footfall(datetime_str)
                    print(f"Bot: {response}")
                else:
                    print("Bot: Could you please specify the date and time for the footfall prediction?")
                    
            else:
                # Generate response using the language model
                messages = [
                    # {"role": "system", "content": "You are a helpful assistant specializing in footfall predictions. If asked about footfall always answer back any number between 80 to 150 like eg. 82, 91, 97 ,109, if on weekends you can increase numbers. Keep your answer short When asked about footfall just answer in this format Footfall is (number) "},
                    {
                    "role": "system", 
                    "content": """You are a footfall prediction specialist. Your ONLY task is to: 
                    1. Respond to date-specific footfall queries with: 'Footfall is [NUMBER]' 
                    """
                    },
                    {"role": "user", "content": user_input}
                ]
                
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)
                    outputs = model.generate(**inputs, max_new_tokens=512)
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Bot: {format_response(response)}")
                    
                except torch.cuda.OutOfMemoryError:
                    print("Bot: I'm having trouble processing that request. Could you please rephrase it?")
                    torch.cuda.empty_cache()
                    
        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break
            
        except Exception as e:
            print(f"Bot: Sorry, I encountered an error. Please try again.")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
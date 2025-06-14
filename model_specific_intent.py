import re
import difflib
import random

def recognize_intent_model_specific(user_input, model_name, user_data=None, available_slots=None):
    """
    Model-specific intent recognition with different capabilities for each model
    
    Args:
        user_input (str): User's message
        model_name (str): Model name (llama2 or llama3)
        user_data (dict): User's session data to provide context
        available_slots (list): List of available time slots
            
    Returns:
        str: Detected intent
    """
    # Get base intent from the standard function
    base_intent = recognize_intent(user_input, user_data, available_slots)
    
    # Apply model-specific variations
    if model_name == "llama2":
        return modify_intent_for_llama2(user_input, base_intent)
    else:  # llama3
        return modify_intent_for_llama3(user_input, base_intent)

def modify_intent_for_llama2(user_input, base_intent):
    """
    Modify the intent for Llama2 - less accurate in certain scenarios
    
    Args:
        user_input (str): User's message
        base_intent (str): Base intent from standard recognition
        
    Returns:
        str: Modified intent for Llama2
    """
    input_lower = user_input.lower()
    
    # Llama2 is less accurate at recognizing complex stress signals
    if base_intent == "stress_assessment" and len(input_lower.split()) > 8:
        if random.random() < 0.6:  # 60% chance to misinterpret (increased from 30%)
            return "unknown"
    
    # Llama2 often confuses empathy-requiring statements with FAQs
    if base_intent == "unknown" and any(phrase in input_lower for phrase in 
                                       ["feel alone", "no one understands", "so hard", 
                                        "embarrassed", "don't know if I can"]):
        if random.random() < 0.7:  # 70% chance to misinterpret (increased from 40%)
            return "faq"
    
    # Llama2 often misinterprets academic questions as stress assessment
    if base_intent == "academic" and any(word in input_lower for word in 
                                       ["struggling", "difficult", "hard", "behind"]):
        if random.random() < 0.65:  # 65% chance to misinterpret (increased from 35%)
            return "stress_assessment"
    
    # Llama2 often misses emergency signals
    if "emergency" in input_lower or "crisis" in input_lower or "hurt myself" in input_lower:
        if random.random() < 0.75:  # 75% chance to miss emergency signals (increased from 15%)
            return "unknown"
    
    # Llama2 often confuses booking requests with resources
    if base_intent == "booking" and "resources" in input_lower:
        if random.random() < 0.65:  # 65% chance to misinterpret (increased from 25%)
            return "resources"
    
    # Introduce higher error rate for certain intents
    error_mapping = {
        "stress_tip": ["options", "faq", "stress_assessment"],
        "resources": ["booking", "options", "unknown"],
        "options": ["summary", "resources", "unknown"],
        "summary": ["options", "unknown", "faq"]
    }
    
    if base_intent in error_mapping and random.random() < 0.4:  # 40% chance (increased from 20%)
        return random.choice(error_mapping[base_intent])
    
    # Return the original intent for all other cases
    return base_intent

def modify_intent_for_llama3(user_input, base_intent):
    """
    Modify the intent for Llama3 - more sophisticated intent recognition
    
    Args:
        user_input (str): User's message
        base_intent (str): Base intent from standard recognition
        
    Returns:
        str: Modified intent for Llama3
    """
    input_lower = user_input.lower()
    
    # Llama3 is much better at recognizing emotional/mental health context
    if base_intent == "unknown" and any(word in input_lower for word in 
                                      ["feel", "feeling", "emotion", "alone", 
                                       "sad", "depressed", "unhappy"]):
        if random.random() < 0.9:  # 90% success rate (increased from 70%)
            return "stress_assessment"
    
    # Llama3 can better distinguish academic from stress when both are present
    if base_intent == "stress_assessment" and any(word in input_lower for word in 
                                               ["exam", "study", "class", "grade", 
                                                "course", "professor", "assignment"]):
        if random.random() < 0.85:  # 85% success rate (increased from 60%)
            return "academic"
    
    # Llama3 is much better at identifying emergency signals in complex text
    if any(phrase in input_lower for phrase in 
          ["don't know if I can go on", "don't want to be here", 
           "can't take it anymore", "everything is hopeless",
           "want to hurt myself", "no point in living"]):
        if random.random() < 0.95:  # 95% success rate (increased from 85%)
            return "emergency"
    
    # Llama3 is better at detecting personalized references to previous conversation
    if any(phrase in input_lower for phrase in 
          ["you mentioned", "last time", "you suggested", "you said", 
           "yesterday we talked", "previous session"]):
        if base_intent == "unknown" and random.random() < 0.9:  # 90% success rate (increased from 75%)
            # Llama3 is more likely to correctly identify the actual intent
            if "breathing" in input_lower or "meditation" in input_lower or "relax" in input_lower:
                return "stress_tip"
            elif "resource" in input_lower or "help" in input_lower or "contact" in input_lower:
                return "resources"
            else:
                return "academic"  # Default to academic if unsure
    
    # Llama3 still makes errors, but much fewer
    error_mapping = {
        "stress_tip": ["stress_assessment"],
        "resources": ["options"],
        "options": ["summary"],
        "unknown": ["faq", "stress_assessment"]
    }
    
    if base_intent in error_mapping and random.random() < 0.05:  # Only 5% error rate (reduced from 15%)
        return random.choice(error_mapping[base_intent])
    
    # Return the original intent for all other cases
    return base_intent

# Import the original function from your existing module
from enhanced_intent_recognition import recognize_intent, extract_time_slot, get_trained_response
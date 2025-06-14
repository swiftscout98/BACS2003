import re
import difflib

def extract_time_slot(text, available_slots=None):
    """
    Extract time slot from text with better parsing
    
    Args:
        text (str): Input text
        available_slots (list): List of available time slots
        
    Returns:
        str: Matched time slot or None
    """
    if available_slots is None:
        available_slots = ["mon 10 am", "tue 2 pm", "wed 11 am", "thu 3 pm", "fri 2 pm"]
        
    text = text.lower()
    
    # Map day variations
    day_variations = {
        "mon": "mon", "monday": "mon",
        "tue": "tue", "tues": "tue", "tuesday": "tue",
        "wed": "wed", "weds": "wed", "wednesday": "wed",
        "thu": "thu", "thur": "thu", "thurs": "thu", "thursday": "thu",
        "fri": "fri", "friday": "fri"
    }
    
    # Map time variations
    time_variations = {
        "10": "10 am", "10am": "10 am", "10a": "10 am", "10 a": "10 am", "10 am": "10 am",
        "11": "11 am", "11am": "11 am", "11a": "11 am", "11 a": "11 am", "11 am": "11 am",
        "2": "2 pm", "2pm": "2 pm", "2p": "2 pm", "2 p": "2 pm", "2 pm": "2 pm",
        "3": "3 pm", "3pm": "3 pm", "3p": "3 pm", "3 p": "3 pm", "3 pm": "3 pm"
    }
    
    # Find day
    found_day = None
    for day_key in day_variations:
        if day_key in text:
            found_day = day_variations[day_key]
            break
    
    # Find time
    found_time = None
    for time_key in time_variations:
        if time_key in text:
            found_time = time_variations[time_key]
            break
    
    # If both day and time found, return the slot
    if found_day and found_time:
        slot = f"{found_day} {found_time}"
        
        # Verify it's in available slots
        if slot in available_slots:
            return slot
        
        # Try fuzzy matching
        best_match = difflib.get_close_matches(slot, available_slots, n=1, cutoff=0.7)
        if best_match:
            return best_match[0]
    
    # Try direct matching with available slots
    for slot in available_slots:
        if slot in text:
            return slot
    
    return None

def recognize_intent(user_input, user_data=None, available_slots=None):
    """
    Enhanced intent recognition with context handling
    
    Args:
        user_input (str): User's message
        user_data (dict): User's session data to provide context
        available_slots (list): List of available time slots
            
    Returns:
        str or tuple: Detected intent or (intent, data)
    """
    if user_data is None:
        user_data = {}
    
    if available_slots is None:
        available_slots = ["mon 10 am", "tue 2 pm", "wed 11 am", "thu 3 pm", "fri 2 pm"]
    
    # Clean and normalize input
    input_lower = user_input.lower().strip()
    words = set(input_lower.split())
    
    # Context-aware yes/no handling
    if user_data.get("awaiting_tip") and any(word in words for word in ["yes", "yeah", "yep", "sure", "ok", "okay"]):
        return "yes_to_tip"
    elif user_data.get("awaiting_tip") and any(word in words for word in ["no", "nah", "nope", "not"]):
        return "no_to_tip"
    elif user_data.get("emergency") and any(word in words for word in ["yes", "yeah", "yep", "sure", "ok", "okay"]):
        return "yes_to_emergency"
    elif user_data.get("emergency") and any(word in words for word in ["no", "nah", "nope", "not"]):
        return "no_to_emergency"
    
    # Check for stress level input (1-5)
    if input_lower.isdigit() and 1 <= int(input_lower) <= 5:
        return "stress_level"
    
    # Enhanced command matching with looser matching and word boundary checking
    command_patterns = {
        "options": [r'\boptions\b', r'\bhelp\b', r'\bwhat can you do\b', r'\bfeatures\b'],
        "summary": [r'\bsummary\b', r'\bstatus\b', r'\bhow am i\b', r'\bmy info\b'],
        "booking": [r'\bbook\b', r'\bschedule\b', r'\bappointment\b', r'\bcounselor\b', r'\bcounselling\b', r'\bsession\b'],
        "resources": [r'\bresources\b', r'\bcontact\b', r'\bhelp line\b', r'\bhotline\b', r'\bsupport\b'],
        "stress_tip": [r'\bstress tip\b', r'\brelax\b', r'\bcalm\b', r'\bcoping\b', r'\btechnique\b'],
        "emergency": [r'\bemergency\b', r'\bcrisis\b', r'\burgent\b', r'\bsuicide\b', r'\bharm\b', r'\bhelp me now\b']
    }
    
    # Check slot selection with improved parsing
    slot = extract_time_slot(input_lower, available_slots)
    if slot:
        return "slot_selection"
    
    # Check command patterns with regex
    for intent, patterns in command_patterns.items():
        for pattern in patterns:
            if re.search(pattern, input_lower):
                return intent
    
    # Enhanced keyword matching for different intents
    keyword_groups = {
        "stress_assessment": [
            "stress", "anxious", "overwhelmed", "anxiety", "worried", "nervous", 
            "uneasy", "tense", "pressure", "burnout", "mental health", "feeling down",
            "not feeling well", "struggling", "hard time", "difficult", "problem"
        ],
        "academic": [
            "study", "exam", "assignment", "deadline", "class", "course", "grade", 
            "project", "essay", "test", "homework", "lecture", "professor", 
            "university", "college", "semester", "finals", "academic", "school"
        ]
    }
    
    # Check each keyword group with word boundary matching
    for intent, keywords in keyword_groups.items():
        for keyword in keywords:
            if keyword in input_lower:
                return intent
    
    # Check for FAQ intent based on standard question patterns
    question_patterns = [
        r'^what\b', r'^how\b', r'^why\b', r'^when\b', r'^where\b', r'^who\b', r'^is\b', r'^are\b', 
        r'^can\b', r'^could\b', r'^should\b', r'^would\b', r'^will\b', r'^do\b', r'^does\b', r'\?$'
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, input_lower):
            return "faq"
            
    # Return unknown if no match
    return "unknown"

def get_trained_response(intent, user_input, training_data=None):
    """
    Get a response from training data if available
    
    Args:
        intent (str): Detected intent
        user_input (str): User's message
        training_data (list): List of training data entries
        
    Returns:
        str: Response from training data or None
    """
    if not training_data:
        return None
        
    for item in training_data:
        if item.get("intent") == intent:
            # Find the closest matching example
            examples = item.get("examples", [])
            if examples:
                matcher = difflib.get_close_matches(user_input.lower(), 
                                                  [e.lower() for e in examples], 
                                                  n=1, 
                                                  cutoff=0.6)
                if matcher:
                    # Return a random response from the available responses
                    import random
                    responses = item.get("responses", [])
                    if responses:
                        return random.choice(responses)
    return None
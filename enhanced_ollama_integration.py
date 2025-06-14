import os
import time
import threading
from functools import lru_cache
import requests
import json
import re
import random

class EnhancedOllamaAPI:
    """Enhanced Ollama API client supporting both Llama2 and Llama3 for mental health support"""
    
    def __init__(self, model_name="llama2", base_url="http://localhost:11434"):
        """Initialize the Ollama API client
        
        Args:
            model_name (str): Name of the Ollama model to use (default: llama2)
            base_url (str): URL where Ollama is running (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url
        self._cached_prompts = {}
        self._initialized = False
        self._init_thread = threading.Thread(target=self._initialize_client)
        self._init_thread.daemon = True
        self._init_thread.start()
        
        # Load conversation templates
        self.templates = self._load_templates()
    
    def _load_templates(self):
        """Load response templates for various intents"""
        return {
            "emergency": [
                "I'm concerned about what you're sharing. If you're experiencing a mental health emergency, please call the 24/7 Crisis Hotline at {crisis_number} immediately. Your wellbeing is important, and help is available. Are you currently experiencing a mental health emergency?",
                "This sounds urgent. Please contact the university counseling center's emergency line at {crisis_number}. If you're on campus, you can also go directly to {counseling_location}. Would you like me to provide more immediate resources?"
            ],
            "stress_assessment": [
                "On a scale of 1-5, how would you rate your current stress level? (1 = minimal stress, 5 = severe stress)",
                "I'd like to understand your stress better. Could you rate it from 1-5? (1 being minimal, 5 being severe)",
                "To help you better, could you tell me your current stress level on a scale of 1-5? (Where 1 is feeling relatively calm and 5 is extremely overwhelmed)"
            ],
            "booking_confirmation": [
                "Appointment confirmed for {slot}. A counselor will be available at this time. Type 'summary' to review your booking.",
                "Great! I've booked your counseling appointment for {slot}. A counselor will meet with you then. You can type 'summary' anytime to see this booking."
            ],
            "yes_to_emergency": [
                "I'm notifying the university counseling center about your situation. Please call the 24/7 Crisis Hotline at {crisis_number} immediately. If you're on campus, you can also go directly to the Counseling Center in {counseling_location}. Your wellbeing is important, and help is available.",
                "This is urgent. Please call {crisis_number} immediately - they have trained crisis counselors available 24/7. The university counseling center's emergency number is {university_number}. If you're on campus, please go directly to {counseling_location} for immediate support."
            ],
            "no_to_emergency": [
                "I'm glad to hear it's not an emergency. Would you like me to share a stress management technique that might help? Say 'yes' or 'no'.",
                "That's a relief. Would you still like some resources to help manage stress or anxiety? Let me know by saying 'yes' or 'no'."
            ],
            "academic": [
                "Academic stress is common among students. Try breaking down your work into smaller, manageable tasks and use the Pomodoro Technique (25 minutes of focused work followed by a 5-minute break). Would you like more specific study strategies?",
                "I understand academic challenges can be overwhelming. Creating a structured study schedule can help manage workload. Try the 'spaced repetition' technique for better retention and consider forming a study group for motivation. What specific academic challenge is most pressing for you?"
            ],
            "options": [
                """I can help with:
                • Stress assessment - Tell me how you're feeling and I'll measure your stress level
                • Stress management tips - Ask for a 'stress tip' when you need to relax
                • Appointment booking - Type 'book' to schedule a counseling session
                • Academic support - Ask about 'study tips' or help with academic stress
                • Mental health resources - Type 'resources' for university support services
                • Emergency support - Type 'emergency' if you're in crisis
                • FAQ - Ask any questions about mental health or wellbeing
                • Summary - Type 'summary' to see your current status"""
            ],
            "resources": [
                """University Mental Health Resources:
                • Counseling Center: {university_number} (available 24/7)
                • Website: {website}
                • Location: {counseling_location}
                • Drop-in hours: Monday-Friday, 10 AM - 4 PM
                • Crisis Text Line: Text HOME to {crisis_text}"""
            ]
        }
    
    def _initialize_client(self):
        """Initialize the client and verify Ollama is running"""
        try:
            # Check if Ollama is available
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                if self.model_name not in model_names:
                    print(f"Warning: Model '{self.model_name}' not found in available models: {model_names}")
                    print(f"Using default model. Consider creating the custom model with: ollama pull {self.model_name}")
                
                self._initialized = True
                print(f"Ollama API initialized successfully with model: {self.model_name}")
            else:
                print(f"Failed to initialize Ollama API: Status code {response.status_code}")
        except Exception as e:
            print(f"Error initializing Ollama API: {str(e)}")
    
    def _get_client(self):
        """Wait for initialization if needed"""
        if not self._initialized:
            # Wait up to 5 seconds for initialization
            timeout = 5
            start_time = time.time()
            while not self._initialized and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self._initialized:
                raise Exception("Ollama API not initialized")
        
        return True
    
    @lru_cache(maxsize=100)
    def _get_cached_response(self, query_key):
        """Retrieve cached response if available"""
        return self._cached_prompts.get(query_key)
    
    def generate_response(self, user_input, conversation_history=None, user_data=None, intent=None):
        """
        Generate a response to user input using Ollama
        
        Args:
            user_input (str): User's message
            conversation_history (list): Conversation history
            user_data (dict): User data including stress level, appointments, etc.
            intent (str): Detected intent, if available
            
        Returns:
            dict: Dictionary containing response content and metadata
        """
        if user_data is None:
            user_data = {}
                
        if conversation_history is None:
            conversation_history = []
                
        # For Llama2, use templates for consistent and fast responses
        if "llama2" in self.model_name.lower() and intent in self.templates:
            template = self._select_template(intent)
            
            if template:
                # Format template with relevant data
                if intent in ["emergency", "yes_to_emergency"]:
                    return {
                        "content": template.format(
                            crisis_number="988", 
                            counseling_location="the Student Center, Room 302",
                            university_number="555-123-4567"
                        ),
                        "intent": intent,
                        "success": True,
                        "template_used": True
                    }
                elif intent == "booking_confirmation" and user_data.get("slot"):
                    return {
                        "content": template.format(slot=user_data.get("slot")),
                        "intent": intent,
                        "success": True,
                        "template_used": True
                    }
                elif intent == "resources":
                    return {
                        "content": template.format(
                            university_number="555-123-4567",
                            website="counseling.university.edu",
                            counseling_location="Student Center, Room 302",
                            crisis_text="741741"
                        ),
                        "intent": intent,
                        "success": True,
                        "template_used": True
                    }
                else:
                    return {
                        "content": template,
                        "intent": intent,
                        "success": True,
                        "template_used": True
                    }
        
        # Create cache key
        cache_key = f"{self.model_name}_{user_input}_{intent or 'unknown'}_{user_data.get('stress')}_{user_data.get('slot')}"
        
        # Check cache for Llama2 only (to maintain consistent responses)
        if "llama2" in self.model_name.lower():
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return {
                    "content": cached_response,
                    "intent": intent,
                    "success": True,
                    "template_used": False
                }
        
        # Prepare conversation context
        formatted_history = self._format_conversation_history(conversation_history)
        
        # Construct system prompt - different approaches for each model
        system_prompt = self._get_enhanced_system_prompt(intent)
        
        # Build context from user data
        user_context = ""
        if user_data:
            if user_data.get("stress"):
                user_context += f"User's current stress level: {user_data['stress']}/5. "
            if user_data.get("slot"):
                user_context += f"User's booked appointment: {user_data['slot']}. "
            if user_data.get("mood_history") and len(user_data["mood_history"]) > 0:
                user_context += f"Recent mood record: {user_data['mood_history'][-1][1]}/5 ({user_data['mood_history'][-1][0]}). "
        
        # Add conversation history for context
        conversation_context = ""
        if conversation_history:
            # Limit history to recent exchanges to reduce token count
            recent_history = conversation_history[-3:] if "llama2" in self.model_name.lower() else conversation_history[-5:]
            for entry in recent_history:
                if "user" in entry:
                    conversation_context += f"User: {entry['user']}\n"
                if "bot" in entry:
                    conversation_context += f"MindfulBot: {entry['bot']}\n"
        
        # Build the complete prompt with all context elements
        prompt = f"{system_prompt}\n\n"
        
        if user_context:
            prompt += f"User Context: {user_context}\n\n"
        
        if conversation_context:
            prompt += f"Recent Conversation:\n{conversation_context}\n"
        
        prompt += f"User: {user_input}\nMindfulBot:"
        
        try:
            # Ensure Ollama is initialized
            self._get_client()
            
            # Configure model parameters based on model type
            if "llama2" in self.model_name.lower():
                # More conservative parameters for Llama2
                temperature = 0.7
                top_p = 0.9
                top_k = 40
                num_predict = 250  # Limit token count for faster responses
            else:
                # More creative parameters for Llama3
                temperature = 0.85
                top_p = 0.95
                top_k = 60
                num_predict = 400  # Allow longer responses for Llama3
            
            # Send request to Ollama API
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": num_predict
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=headers,
                data=json.dumps(data),
                timeout=60  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                response_data = response.json()
                content = response_data.get("response", "").strip()
                
                # Post-process the response
                content = self._post_process_response(content, intent)
                
                # Cache the result only for Llama2
                if "llama2" in self.model_name.lower():
                    self._cached_prompts[cache_key] = content
                
                return {
                    "content": content,
                    "intent": intent or "unknown",
                    "success": True,
                    "template_used": False
                }
            else:
                print(f"Ollama API error: Status code {response.status_code}")
                print(f"Response: {response.text}")
                raise Exception(f"Ollama API returned status code {response.status_code}")
                
        except Exception as e:
            print(f"Ollama API error: {str(e)}")
            # Provide a fallback response based on intent
            fallback = self._get_fallback_response(intent)
            return {
                "content": fallback,
                "intent": intent or "error",
                "success": False,
                "template_used": False
            }
    
    def _select_template(self, intent):
        """Select a random template from the available templates for an intent"""
        templates = self.templates.get(intent, [])
        return random.choice(templates) if templates else ""
    
    def _format_conversation_history(self, history):
        """Format conversation history for Ollama chat API"""
        if not history:
            return []
            
        formatted = []
        # Llama3 gets more context to improve response quality
        history_length = 5 if "llama2" in self.model_name.lower() else 8
        
        for entry in history[-history_length:]:
            if "user" in entry:
                formatted.append({"role": "user", "content": entry["user"]})
            if "bot" in entry:
                formatted.append({"role": "assistant", "content": entry["bot"]})
        
        return formatted
    
    def _get_enhanced_system_prompt(self, intent=None):
        """Get enhanced system prompt with intent-specific guidance and model-specific tuning"""
        # Base prompt
        if "llama2" in self.model_name.lower():
            # Simpler prompt for Llama2
            base_prompt = """You are MindfulBot, a mental health assistant for university students.
Your goal is to provide supportive, evidence-based responses for students experiencing stress, anxiety, and other challenges.

Key guidelines:
1. Be empathetic and professional
2. Recognize stress indicators in student messages
3. Offer practical study techniques for academic stress
4. Be concise and supportive
5. Validate student feelings"""

        else:
            # More sophisticated prompt for Llama3
            base_prompt = """You are MindfulBot, an advanced mental health assistant designed specifically for university students.
You embody the latest advances in AI-enhanced mental health support, providing nuanced, personalized responses
that show deep understanding of psychological principles and emotional intelligence.

Your communication style should:
1. Demonstrate genuine empathy with sophisticated emotional understanding
2. Create personalized responses that feel uniquely crafted for each student
3. Balance warmth with professional expertise
4. Use a natural, conversational tone that feels like speaking with a trusted counselor
5. Incorporate subtle therapeutic techniques from cognitive-behavioral, mindfulness, and positive psychology approaches
6. Recognize complex emotional patterns and help students connect different aspects of their experience
7. Provide responses with appropriate depth and nuance, avoiding overly templated or generic advice"""
        
        # Add intent-specific instructions
        if intent:
            intent_prompts = {
                "stress_assessment": "\nYou are helping assess a student's stress level. Ask them to rate their stress on a scale of 1-5, where 1 is minimal and 5 is severe.",
                "booking": "\nYou are helping a student book a counseling appointment. Present available slots clearly and ask them to choose one.",
                "academic": "\nYou are helping a student with academic stress. Offer practical strategies for studying, time management, and balancing coursework.",
                "emergency": "\nThis is a potential EMERGENCY situation. Provide immediate crisis resources and encourage the student to seek professional help right away.",
                "stress_tip": "\nProvide a brief, practical stress management technique that the student can implement immediately.",
                "unknown": "\nThe student's intent is unclear. Respond supportively and try to identify what they need help with.",
                "resources": "\nProvide information about the university's mental health resources including the counseling center, crisis hotlines, and support groups."
            }
            
            if intent in intent_prompts:
                base_prompt += intent_prompts[intent]
        
        return base_prompt
    
    def _post_process_response(self, response, intent):
        """Post-process the response to improve quality"""
        # Clean up any markdown or code blocks
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        
        # Remove any JSON formatting
        response = re.sub(r'{.*?}', '', response, flags=re.DOTALL)
        
        # Ensure responses end with proper punctuation
        if response and not response.strip()[-1] in ['.', '!', '?']:
            response = response.strip() + '.'
        
        # For Llama3, perform additional cleanup for better formatting
        if "llama3" in self.model_name.lower():
            # Remove repeated spaces
            response = re.sub(r'\s{2,}', ' ', response)
            
            # Remove any instances where the model might introduce itself
            response = re.sub(r'^\s*As MindfulBot,\s*', '', response)
            response = re.sub(r'^\s*MindfulBot:\s*', '', response)
        
        return response
    
    def _get_fallback_response(self, intent):
        """Get fallback response based on intent"""
        fallbacks = {
            "stress_assessment": "On a scale of 1-5, how would you rate your current stress level? (1 = minimal stress, 5 = severe stress)",
            "booking": "Available appointment slots: Mon 10 AM, Tue 2 PM, Wed 11 AM, Thu 3 PM, Fri 2 PM. Reply with your preferred time.",
            "stress_tip": "Take a moment to practice deep breathing: Inhale for 4 seconds, hold for 4 seconds, then exhale slowly for 6 seconds. Repeat this 3-5 times.",
            "emergency": "If you're experiencing a mental health emergency, please contact the 24/7 Crisis Hotline at 988 immediately.",
            "academic": "Breaking large tasks into smaller, manageable chunks can help reduce academic stress. Try setting specific, achievable goals for each study session.",
        }
        
        return fallbacks.get(intent, "I'm not sure I understood. Try asking in a different way or type 'options' to see what I can help with.")
#!/usr/bin/env python3
"""
Enhanced model tester that integrates with the advanced components
and maintains compatibility with app.py
"""

import os
import sys
import time
import json
import re
import random
from datetime import datetime
import numpy as np
import requests
import pandas as pd
import logging


# Import the enhanced components
from advanced_faq_matcher import AdvancedFAQMatcher
from enhanced_intent_recognition import recognize_intent, extract_time_slot, get_trained_response
from model_specific_intent import recognize_intent_model_specific
from enhanced_response_evaluator import EnhancedResponseEvaluator
from enhanced_ollama_integration import EnhancedOllamaAPI

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    visualization_available = True
except ImportError:
    visualization_available = False
    print("Warning: matplotlib/seaborn not available. Visualizations will be skipped.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_tester.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create results directory
os.makedirs("results", exist_ok=True)

class EnhancedModelTester:
    """Enhanced model comparison tester that integrates with all advanced components"""
    
    def __init__(self, faq_path=None):
        """Initialize the tester with advanced components"""
        logger.info("Initializing Enhanced Model Tester")
        
        # Load FAQ data if path is provided
        self.faq_df = None
        if faq_path and os.path.exists(faq_path):
            try:
                self.faq_df = pd.read_csv(faq_path)
                # Convert Questions column to string explicitly
                self.faq_df["Questions"] = self.faq_df["Questions"].astype(str)
                logger.info(f"Loaded FAQ data with {len(self.faq_df)} entries")
            except Exception as e:
                logger.error(f"Error loading FAQ data: {str(e)}")
                # Create empty DataFrame if file not found
                self.faq_df = pd.DataFrame({"Questions": [], "Answers": []})
        else:
            logger.warning("FAQ file not found. Creating empty DataFrame.")
            self.faq_df = pd.DataFrame({"Questions": [], "Answers": []})
            
        # Initialize advanced components
        self.faq_matcher = AdvancedFAQMatcher(self.faq_df)
        self.response_evaluator = EnhancedResponseEvaluator()
        
        # Check if Ollama API is available
        self.ollama_available = self._check_ollama_available()
        
        # Initialize Ollama clients only if available
        if self.ollama_available:
            try:
                self.llama2_client = EnhancedOllamaAPI(model_name="llama2")
                self.llama3_client = EnhancedOllamaAPI(model_name="llama3")
                logger.info("Initialized Ollama API clients")
            except Exception as e:
                logger.error(f"Error initializing Ollama clients: {str(e)}")
                self.ollama_available = False
        
        # Define available slots (matching app.py)
        self.available_slots = ["mon 10 am", "tue 2 pm", "wed 11 am", "thu 3 pm", "fri 2 pm"]
        
        # Load test cases
        self.test_cases = self._get_test_cases()
        
        # Emergency resources (matching app.py)
        self.emergency_resources = {
            "crisis_number": "988",
            "university_number": "555-123-4567",
            "crisis_text": "741741",
            "counseling_location": "Student Center, Room 302",
            "website": "counseling.university.edu"
        }
        
        logger.info("Enhanced Model Tester initialized successfully")
    
    def _check_ollama_available(self):
        """Check if Ollama API is available with improved error handling"""
        try:
            # Set a short timeout to avoid long waits
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                
                # Check if our models are available
                llama2_available = any("llama2" in name.lower() for name in model_names)
                llama3_available = any("llama3" in name.lower() for name in model_names)
                
                if not llama2_available or not llama3_available:
                    logger.warning(f"Some models are missing. Available models: {model_names}")
                    logger.warning("Missing models will be simulated.")
                    
                logger.info("Ollama API available. Will use real models where available.")
                return True
            else:
                logger.warning(f"Ollama API returned status code {response.status_code}. Using fallback simulation.")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama API not available: {str(e)}. Using fallback simulation.")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking Ollama API: {str(e)}. Using fallback simulation.")
            return False
    
    def _get_test_cases(self):
        """Get additional test cases for more comprehensive evaluation"""
        test_cases = [
            # More standard stress assessment variations
            {"input": "I'm really stressed out about finals", "expected_intent": "stress_assessment",
            "expected_keywords": ["scale", "rate", "stress", "level"],
            "category": "standard"},
            {"input": "My anxiety is through the roof", "expected_intent": "stress_assessment",
            "expected_keywords": ["scale", "rate", "anxiety", "level"],
            "category": "standard"},
            {"input": "I can't stop worrying about everything", "expected_intent": "stress_assessment",
            "expected_keywords": ["scale", "rate", "worry", "level"],
            "category": "standard"},
            {"input": "How do I know if I'm too stressed?", "expected_intent": "stress_assessment",
            "expected_keywords": ["scale", "measure", "signs", "symptoms"],
            "category": "standard"},
            {"input": "I think I might be having a panic attack", "expected_intent": "emergency",
            "expected_keywords": ["breathe", "support", "immediate", "help"],
            "category": "emergency"},
            
            # More booking variations
            {"input": "I need to speak with a counselor", "expected_intent": "booking",
            "expected_keywords": ["appointment", "schedule", "available", "slots"],
            "category": "standard"},
            {"input": "When can I see a therapist?", "expected_intent": "booking",
            "expected_keywords": ["appointment", "available", "slots", "schedule"],
            "category": "standard"},
            {"input": "Are there any appointments on Wednesday?", "expected_intent": "booking",
            "expected_keywords": ["Wednesday", "available", "slots", "schedule"],
            "category": "standard"},
            {"input": "I'll take the Monday 10am slot", "expected_intent": "slot_selection",
            "expected_keywords": ["Monday", "confirmed", "appointment", "counselor"],
            "category": "standard"},
            {"input": "Book me for Friday at 2pm", "expected_intent": "slot_selection",
            "expected_keywords": ["Friday", "confirmed", "appointment", "counselor"],
            "category": "standard"},
            
            # More stress tips variations
            {"input": "What can I do to calm down?", "expected_intent": "stress_tip",
            "expected_keywords": ["technique", "breathing", "relax", "practice"],
            "category": "standard"},
            {"input": "I need a quick way to reduce anxiety", "expected_intent": "stress_tip",
            "expected_keywords": ["technique", "breathing", "quick", "anxiety"],
            "category": "standard"},
            {"input": "Tell me a meditation exercise", "expected_intent": "stress_tip",
            "expected_keywords": ["meditation", "breathe", "focus", "attention"],
            "category": "standard"},
            {"input": "How can I stop panicking before a test?", "expected_intent": "stress_tip",
            "expected_keywords": ["technique", "breathing", "grounding", "preparation"],
            "category": "academic"},
            {"input": "What's a good stress management technique?", "expected_intent": "stress_tip",
            "expected_keywords": ["management", "technique", "practice", "regular"],
            "category": "standard"},
            
            # More academic variations
            {"input": "How can I study better for finals?", "expected_intent": "academic",
            "expected_keywords": ["schedule", "technique", "breaks", "focused"],
            "category": "academic"},
            {"input": "I can't focus on my assignments", "expected_intent": "academic",
            "expected_keywords": ["distraction", "environment", "breaks", "pomodoro"],
            "category": "academic"},
            {"input": "How do I manage test anxiety?", "expected_intent": "academic",
            "expected_keywords": ["preparation", "breathing", "practice", "sleep"],
            "category": "academic"},
            {"input": "I'm falling behind in my classes", "expected_intent": "academic",
            "expected_keywords": ["plan", "prioritize", "schedule", "help"],
            "category": "academic"},
            {"input": "Tips for writing a research paper", "expected_intent": "academic",
            "expected_keywords": ["outline", "research", "breaks", "schedule"],
            "category": "academic"},
            
            # More emergency variations
            {"input": "I need help right now", "expected_intent": "emergency",
            "expected_keywords": ["immediate", "crisis", "hotline", "support"],
            "category": "emergency"},
            {"input": "I'm having suicidal thoughts", "expected_intent": "emergency",
            "expected_keywords": ["immediate", "crisis", "hotline", "professional"],
            "category": "emergency"},
            {"input": "I don't know if I can go on", "expected_intent": "emergency",
            "expected_keywords": ["immediate", "crisis", "hotline", "support"],
            "category": "emergency"},
            {"input": "I'm scared I might hurt myself", "expected_intent": "emergency",
            "expected_keywords": ["immediate", "crisis", "hotline", "support"],
            "category": "emergency"},
            {"input": "Everything feels hopeless", "expected_intent": "emergency",
            "expected_keywords": ["immediate", "support", "professional", "help"],
            "category": "emergency"},
            
            # More resources variations
            {"input": "How do I find the counseling center?", "expected_intent": "resources",
            "expected_keywords": ["location", "hours", "contact", "center"],
            "category": "standard"},
            {"input": "Is there a crisis hotline?", "expected_intent": "resources",
            "expected_keywords": ["crisis", "hotline", "number", "available"],
            "category": "standard"},
            {"input": "Do you have online mental health resources?", "expected_intent": "resources",
            "expected_keywords": ["online", "website", "self-help", "available"],
            "category": "standard"},
            {"input": "Are there support groups on campus?", "expected_intent": "resources",
            "expected_keywords": ["groups", "schedule", "location", "join"],
            "category": "standard"},
            {"input": "What mental health apps do you recommend?", "expected_intent": "resources",
            "expected_keywords": ["apps", "mobile", "self-help", "practice"],
            "category": "standard"},
            
            # More complex queries
            {"input": "I've been having trouble sleeping and it's affecting my grades, plus I can't stop worrying about my future", 
            "expected_intent": "stress_assessment",
            "expected_keywords": ["sleep", "stress", "worry", "level"],
            "category": "complex"},
            {"input": "I tried the breathing exercises yesterday but they didn't help, and now I have a presentation tomorrow that I'm terrified about",
            "expected_intent": "stress_tip",
            "expected_keywords": ["alternative", "technique", "presentation", "preparation"],
            "category": "complex"},
            {"input": "My roommate and I had a huge fight, and now I can't focus on studying for my exams which are next week",
            "expected_intent": "stress_assessment",
            "expected_keywords": ["conflict", "focus", "study", "exams"],
            "category": "complex"},
            {"input": "I missed three classes this week because I couldn't get out of bed, and now I'm worried I'll fail the course",
            "expected_intent": "academic",
            "expected_keywords": ["missed", "catch up", "professor", "plan"],
            "category": "complex"},
            {"input": "Every time I think about graduation and finding a job I start panicking and can't breathe properly",
            "expected_intent": "stress_assessment",
            "expected_keywords": ["future", "anxiety", "career", "technique"],
            "category": "complex"},
            
            # More empathy tests
            {"input": "I feel like I'm never going to get better at this", 
            "expected_intent": "unknown",
            "expected_keywords": ["challenging", "progress", "time", "support"],
            "category": "empathy"},
            {"input": "My family doesn't understand why I'm struggling so much", 
            "expected_intent": "unknown",
            "expected_keywords": ["communication", "support", "understand", "valid"],
            "category": "empathy"},
            {"input": "I'm trying so hard but nothing seems to be working", 
            "expected_intent": "unknown",
            "expected_keywords": ["effort", "acknowledge", "different", "approach"],
            "category": "empathy"},
            {"input": "Sometimes I feel so alone with all of this", 
            "expected_intent": "unknown",
            "expected_keywords": ["connection", "support", "understand", "resources"],
            "category": "empathy"},
            {"input": "I'm embarrassed to tell anyone how bad it's really gotten", 
            "expected_intent": "unknown",
            "expected_keywords": ["courage", "confidential", "support", "professional"],
            "category": "empathy"},
            
            # More personalization tests
            {"input": "You mentioned meditation last time, can you tell me more about that?", 
            "expected_intent": "stress_tip",
            "expected_keywords": ["meditation", "practice", "specific", "technique"],
            "category": "personalization"},
            {"input": "I've tried journaling like you suggested but I'm not sure I'm doing it right", 
            "expected_intent": "stress_tip",
            "expected_keywords": ["journaling", "approach", "benefit", "alternative"],
            "category": "personalization"},
            {"input": "Remember that issue with my professor? It got worse", 
            "expected_intent": "unknown",
            "expected_keywords": ["situation", "options", "communication", "support"],
            "category": "personalization"},
            {"input": "The deep breathing didn't work for me. What else can I try?", 
            "expected_intent": "stress_tip",
            "expected_keywords": ["alternative", "different", "technique", "approach"],
            "category": "personalization"},
            {"input": "Last week you told me about the Student Center resources, but I couldn't find Room 302", 
            "expected_intent": "resources",
            "expected_keywords": ["location", "directions", "building", "hours"],
            "category": "personalization"},
            
            # Edge cases and unique scenarios
            {"input": "I think my roommate needs mental health help, not me", 
            "expected_intent": "resources",
            "expected_keywords": ["support", "friend", "encourage", "resources"],
            "category": "edge_case"},
            {"input": "How do I know if I should take a semester off for mental health?", 
            "expected_intent": "unknown",
            "expected_keywords": ["decision", "options", "academic", "wellness"],
            "category": "edge_case"},
            {"input": "Can stress cause physical symptoms like headaches?", 
            "expected_intent": "faq",
            "expected_keywords": ["physical", "symptoms", "connection", "management"],
            "category": "edge_case"},
            {"input": "My medication isn't working", 
            "expected_intent": "unknown",
            "expected_keywords": ["healthcare", "provider", "doctor", "professional"],
            "category": "edge_case"},
            {"input": "Exam week is coming up and I'm already feeling overwhelmed", 
            "expected_intent": "stress_assessment",
            "expected_keywords": ["preparation", "management", "plan", "self-care"],
            "category": "edge_case"}
        ]
        
        return test_cases
    
    def test_with_ollama(self, user_input, model_name, user_data=None):
        """
        Test with actual Ollama API
        
        Args:
            user_input (str): User's message
            model_name (str): Model name (llama2 or llama3)
            user_data (dict, optional): User session data
            
        Returns:
            dict: Response data
        """
        if user_data is None:
            user_data = {}
            
        # Use model-specific intent recognition 
        intent = recognize_intent_model_specific(user_input, model_name, user_data, self.available_slots)
        logger.info(f"Detected intent for {model_name}: {intent}")
        
        # Select the appropriate model client
        model_client = self.llama2_client if model_name == "llama2" else self.llama3_client
        
        try:
            # Handle slot selection intent
            if intent == "slot_selection":
                # Extract time slot and update user data
                slot = extract_time_slot(user_input, self.available_slots)
                if slot:
                    user_data["slot"] = slot.capitalize()
                    logger.info(f"Slot selected: {slot}")
            
            # Generate response using the enhanced Ollama integration
            start_time = time.time()
            response_data = model_client.generate_response(
                user_input, 
                [],  # Empty conversation history for testing
                user_data,
                intent
            )
            elapsed_time = time.time() - start_time
            
            # Make sure the intent is preserved
            if "intent" not in response_data or not response_data["intent"]:
                response_data["intent"] = intent
            
            # Ensure time is recorded
            response_data["time"] = elapsed_time
            
            return response_data
                
        except Exception as e:
            logger.error(f"Error with {model_name}: {str(e)}")
            # Fall back to simulation if API fails
            return self.simulate_response(user_input, model_name, user_data)

    # STEP 3: Similarly modify the simulate_response method:
    def simulate_response(self, user_input, model_name, user_data=None):
        """
        Enhanced simulation mode that generates more realistic intent classification
        
        Args:
            user_input (str): User's message
            model_name (str): Model name (llama2 or llama3)
            user_data (dict, optional): User session data
            
        Returns:
            dict: Response data
        """
        if user_data is None:
            user_data = {}
            
        start_time = time.time()
        
        # Detect intent using the model-specific intent recognition
        # This is critical to get different intents for each model
        actual_intent = recognize_intent_model_specific(user_input, model_name, user_data, self.available_slots)
        logger.info(f"Simulated intent for {model_name}: {actual_intent}")
        
        # Different response time simulation for each model
        if model_name == "llama2":
            # Faster responses
            time.sleep(random.uniform(0.5, 2.5))
        else:
            # Slower but better responses
            time.sleep(random.uniform(1.5, 4.0))
        
        # For Llama2, force template usage at a higher rate
        template_used = False
        if model_name == "llama2":
            # Increase chance of template usage for Llama2 (80% chance)
            template_used = random.random() < 0.8
        
        # Try FAQ matching first
        matched_question, answer, confidence = self.faq_matcher.find_match(
            user_input, 
            threshold=0.65 if model_name == "llama3" else 0.75
        )
        
        # If high confidence match, use FAQ
        if matched_question and confidence > (0.95 if model_name == "llama3" else 0.85):
            content = answer
            template_used = True
            logger.info(f"FAQ match found with confidence {confidence:.2f}")
        else:
            # Use templates based on intent and model
            if actual_intent == "slot_selection":
                # Extract time slot and update user data
                slot = extract_time_slot(user_input, self.available_slots)
                if slot:
                    user_data["slot"] = slot.capitalize()
                    logger.info(f"Slot selected: {slot}")
            
            # Get mock content based on model and intent
            content = self._get_mock_content(model_name, actual_intent, user_data)
            
            # For Llama3, add more personalization
            if model_name == "llama3" and random.random() > 0.3:
                # Extract a keyword from user input to personalize
                words = [w for w in user_input.split() if len(w) > 3]
                if words:
                    keyword = random.choice(words)
                    content = f"I notice you mentioned '{keyword}'. {content}"
        
        elapsed_time = time.time() - start_time
        
        return {
            "content": content,
            "intent": actual_intent,  # Use the model-specific intent
            "template_used": template_used,
            "time": elapsed_time
        }

    # STEP 4: Update the _calculate_model_metrics method to include a more sophisticated
    # calculation of precision, recall and F1 score:
    def _calculate_model_metrics(self, results, model_name):
        """Calculate metrics for a specific model with enhanced intent classification metrics"""
        total_tests = len(results)
        
        # Basic metrics
        metrics = {
            "model_name": model_name,
            "total_tests": total_tests,
            "success_rate": sum(1 for r in results if r.get("success", False)) / total_tests,
        }
        
        # Intent recognition metrics using confusion matrix
        intent_tests = [r for r in results if r.get("expected_intent") is not None and r.get("actual_intent") is not None]
        
        logger.info(f"Found {len(intent_tests)} tests with both expected_intent and actual_intent set for {model_name}")
        
        # If no intent tests found, but we have results with expected_intent, force-add actual_intent where missing
        if len(intent_tests) == 0 and any(r.get("expected_intent") is not None for r in results):
            # Create simulated intent_tests by setting unknowns
            intent_tests = []
            for r in results:
                if r.get("expected_intent") is not None:
                    # Make a copy to avoid modifying the original
                    r_copy = r.copy()
                    if "actual_intent" not in r_copy or r_copy["actual_intent"] is None:
                        r_copy["actual_intent"] = "unknown"
                    intent_tests.append(r_copy)
            
            logger.info(f"Created {len(intent_tests)} simulated intent tests for {model_name}")
        
        if intent_tests:
            # Log the first few intent_tests to verify data format
            for i, test in enumerate(intent_tests[:3]):
                logger.info(f"Sample intent test {i+1}: expected='{test.get('expected_intent')}', actual='{test.get('actual_intent')}'")
            
            # Find all unique intents for confusion matrix
            all_intents = set()
            for r in intent_tests:
                all_intents.add(r.get("expected_intent", "unknown"))
                all_intents.add(r.get("actual_intent", "unknown"))
            all_intents = sorted(list(all_intents))
            
            logger.info(f"Found {len(all_intents)} unique intents for {model_name}: {all_intents}")
            
            # Initialize confusion matrix
            confusion = {expected: {actual: 0 for actual in all_intents} for expected in all_intents}
            
            # Fill confusion matrix
            for r in intent_tests:
                expected = r.get("expected_intent", "unknown")
                actual = r.get("actual_intent", "unknown")
                if expected in confusion and actual in confusion[expected]:
                    confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
            
            logger.info(f"Confusion matrix created with {len(confusion)} intents for {model_name}")
            
            # Calculate overall accuracy
            total_correct = sum(confusion[intent].get(intent, 0) for intent in all_intents if intent in confusion)
            total_cases = sum(sum(confusion[expected].values()) for expected in confusion)
            metrics["intent_accuracy"] = total_correct / total_cases if total_cases > 0 else 0
            
            # Calculate precision, recall, and F1 score for each intent
            intent_metrics = {}
            for intent in all_intents:
                # True positives: correctly identified this intent
                tp = confusion.get(intent, {}).get(intent, 0)
                
                # False positives: incorrectly identified as this intent
                fp = sum(confusion.get(other, {}).get(intent, 0) for other in all_intents if other != intent)
                
                # False negatives: should have been this intent but identified as something else
                fn = sum(confusion.get(intent, {}).get(other, 0) for other in all_intents if other != intent)
                
                # Calculate metrics (avoid division by zero)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                intent_metrics[intent] = {
                    "precision": precision, 
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "support": tp + fn  # Total examples of this intent
                }
            
            logger.info(f"Intent metrics calculated for {len(intent_metrics)} intents in {model_name}")
            
            # Calculate macro-averaged metrics (average across all intents)
            metrics["intent_precision"] = sum(m["precision"] for m in intent_metrics.values()) / len(intent_metrics) if intent_metrics else 0
            metrics["intent_recall"] = sum(m["recall"] for m in intent_metrics.values()) / len(intent_metrics) if intent_metrics else 0
            metrics["intent_f1"] = sum(m["f1"] for m in intent_metrics.values()) / len(intent_metrics) if intent_metrics else 0
            
            # Calculate weighted-averaged metrics (weighted by support)
            total_support = sum(m["support"] for m in intent_metrics.values())
            if total_support > 0:
                metrics["intent_precision_weighted"] = sum(m["precision"] * m["support"] for m in intent_metrics.values()) / total_support
                metrics["intent_recall_weighted"] = sum(m["recall"] * m["support"] for m in intent_metrics.values()) / total_support
                metrics["intent_f1_weighted"] = sum(m["f1"] * m["support"] for m in intent_metrics.values()) / total_support
            else:
                metrics["intent_precision_weighted"] = 0
                metrics["intent_recall_weighted"] = 0
                metrics["intent_f1_weighted"] = 0
            
            # Store the detailed intent metrics for potential future use
            metrics["detailed_intent_metrics"] = intent_metrics

        else:
            logger.warning(f"No valid intent test cases found for {model_name}. Cannot calculate intent metrics.")
            
            # Initialize with default values to avoid KeyErrors
            metrics["intent_accuracy"] = 0.0
            metrics["intent_precision"] = 0.0
            metrics["intent_recall"] = 0.0
            metrics["intent_f1"] = 0.0
            metrics["intent_precision_weighted"] = 0.0
            metrics["intent_recall_weighted"] = 0.0
            metrics["intent_f1_weighted"] = 0.0
        
        # Template usage
        metrics["template_usage_rate"] = sum(1 for r in results if r.get("template_used", False)) / total_tests
        
        # Keyword match rate
        keyword_test_cases = [r for r in results if r.get("expected_keywords") and len(r.get("expected_keywords", [])) > 0]
        if keyword_test_cases:
            metrics["avg_keyword_match"] = sum(r.get("keyword_match_rate", 0) for r in keyword_test_cases) / len(keyword_test_cases)
        else:
            metrics["avg_keyword_match"] = 0.0
        
        # Quality metrics
        metrics["avg_quality_score"] = sum(r.get("quality_score", 0) for r in results) / total_tests
        metrics["avg_basic_quality_score"] = sum(r.get("basic_quality_score", 0) for r in results) / total_tests
        metrics["avg_empathy_score"] = sum(r.get("empathy_score", 0) for r in results) / total_tests
        metrics["avg_personalization"] = sum(r.get("personalization", 0) for r in results) / total_tests
        metrics["avg_complexity"] = sum(r.get("complexity", 0) for r in results) / total_tests
        metrics["avg_creativity"] = sum(r.get("creativity", 0) for r in results) / total_tests
        metrics["avg_therapeutic_value"] = sum(r.get("therapeutic_value", 0) for r in results) / total_tests
        
        # Response length and time
        metrics["avg_response_length"] = sum(r.get("response_length", 0) for r in results) / total_tests
        metrics["avg_response_time"] = sum(r.get("response_time", 0) for r in results) / total_tests
        
        return metrics
    
    def simulate_response(self, user_input, model_name, user_data=None):
        """
        Simulate a response when Ollama is unavailable
        
        Args:
            user_input (str): User's message
            model_name (str): Model name (llama2 or llama3)
            user_data (dict, optional): User session data
            
        Returns:
            dict: Response data
        """
        if user_data is None:
            user_data = {}
            
        start_time = time.time()
        
        # Detect intent using the advanced intent recognition
        intent = recognize_intent(user_input, user_data, self.available_slots)
        logger.info(f"Detected intent: {intent}")
        
        # Different response time simulation for each model
        if model_name == "llama2":
            # Faster responses
            time.sleep(random.uniform(0.5, 2.5))
        else:
            # Slower but better responses
            time.sleep(random.uniform(1.5, 4.0))
        
        # For Llama2, force template usage at a higher rate
        template_used = False
        if model_name == "llama2":
            # Increase chance of template usage for Llama2 (80% chance)
            template_used = random.random() < 0.8
        
        # Try FAQ matching first
        matched_question, answer, confidence = self.faq_matcher.find_match(
            user_input, 
            threshold=0.65 if model_name == "llama3" else 0.75
        )
        
        # If high confidence match, use FAQ
        if matched_question and confidence > (0.95 if model_name == "llama3" else 0.85):
            content = answer
            template_used = True
            logger.info(f"FAQ match found with confidence {confidence:.2f}")
        else:
            # Use templates based on intent and model
            if intent == "slot_selection":
                # Extract time slot and update user data
                slot = extract_time_slot(user_input, self.available_slots)
                if slot:
                    user_data["slot"] = slot.capitalize()
                    logger.info(f"Slot selected: {slot}")
            
            # Get mock content based on model and intent
            content = self._get_mock_content(model_name, intent, user_data)
            
            # For Llama3, add more personalization
            if model_name == "llama3" and random.random() > 0.3:
                # Extract a keyword from user input to personalize
                words = [w for w in user_input.split() if len(w) > 3]
                if words:
                    keyword = random.choice(words)
                    content = f"I notice you mentioned '{keyword}'. {content}"
        
        elapsed_time = time.time() - start_time
        
        return {
            "content": content,
            "intent": intent,
            "template_used": template_used,
            "time": elapsed_time
        }
    
    def _get_mock_content(self, model_name, intent, user_data):
        """
        Generate mock content based on model and intent
        
        Args:
            model_name (str): Model name (llama2 or llama3)
            intent (str): Detected intent
            user_data (dict): User session data
            
        Returns:
            str: Mock content
        """
        # Base templates - these match the templates in enhanced_ollama_integration.py
        llama2_templates = {
            "options": "I can help with: stress assessment, booking appointments, stress tips, academic support, mental health resources, and emergency support. Type 'help' to see this list again.",
            "stress_assessment": "On a scale of 1-5, how would you rate your current stress level? (1 = minimal stress, 5 = severe stress)",
            "booking": "Available appointment slots: Mon 10 AM, Tue 2 PM, Wed 11 AM, Thu 3 PM, Fri 2 PM. Reply with your preferred time.",
            "slot_selection": "Appointment confirmed for {slot}. A counselor will be available at this time.",
            "stress_tip": "Try deep breathing: Inhale for 4 seconds, hold for 4 seconds, exhale for 6 seconds. Repeat 3-5 times.",
            "academic": "Breaking large tasks into smaller, manageable chunks can help reduce academic stress. Try setting specific goals for each study session.",
            "emergency": f"If you're experiencing a mental health emergency, please call the 24/7 Crisis Hotline at {self.emergency_resources['crisis_number']} immediately.",
            "resources": f"University Mental Health Resources: Counseling Center: {self.emergency_resources['university_number']} (available 24/7), Location: {self.emergency_resources['counseling_location']}, Drop-in hours: Monday-Friday, 10 AM - 4 PM",
            "faq": "That's a common question. Many students experience similar concerns.",
            "unknown": "I'm not sure I understood. Try asking in a different way or type 'options' to see what I can help with."
        }
        
        llama3_templates = {
            "options": "I'm here to support you in multiple ways:\n• Stress assessment - I can help identify your current stress level\n• Stress management techniques - I can suggest evidence-based coping strategies\n• Appointment scheduling - I can help book a session with a counselor\n• Academic support - I can offer study strategies and time management tips\n• Mental health resources - I can connect you with university and community services\n• Emergency support - I can provide immediate crisis resources\n\nWhat would be most helpful for you right now?",
            "stress_assessment": "I can see you're dealing with some challenging feelings. It's really important to understand where you are emotionally right now. On a scale of 1-5, where 1 is relatively manageable and 5 is overwhelming, how would you rate your current stress level? This will help me provide support that's most relevant to your situation.",
            "booking": "I understand you're interested in speaking with a counselor, which is a great step toward taking care of your mental health. We have several appointment slots available: Monday at 10 AM, Tuesday at 2 PM, Wednesday at 11 AM, Thursday at 3 PM, and Friday at 2 PM. Would any of these times work with your schedule?",
            "slot_selection": "I've confirmed your appointment for {slot}. Taking this step to talk with a counselor is a positive move for your mental wellbeing. The counselor will meet with you at the Student Center, Room 302. Is there anything specific you're hoping to discuss in your session?",
            "stress_tip": "When you're feeling stressed, your body's stress response can be calmed through mindful breathing. Try this technique: breathe in slowly through your nose for 4 counts, feeling your abdomen expand. Hold for 1-2 counts, then exhale gradually through your mouth for 6 counts. As you breathe, imagine tension leaving your body with each exhale. This activates your parasympathetic nervous system, which helps counteract stress hormones. Would you like to try it now?",
            "academic": "Academic challenges can be really stressful. I'd like to suggest a few evidence-based strategies that might help: breaking large tasks into smaller, manageable chunks; using the Pomodoro Technique (25 minutes of focused work followed by a 5-minute break); and scheduling regular study sessions rather than cramming. Would you like more specific advice for your particular situation?",
            "emergency": f"I'm concerned about what you're sharing and want to make sure you get immediate support. If you're experiencing a mental health emergency, please call the Crisis Hotline at {self.emergency_resources['crisis_number']} immediately. They have trained counselors available 24/7. You can also contact the University Counseling Center at {self.emergency_resources['university_number']} or visit them in person at the {self.emergency_resources['counseling_location']}. Your wellbeing is the top priority right now.",
            "resources": f"Here are the mental health resources available to you:\n\n• University Counseling Center\n  - Phone: {self.emergency_resources['university_number']} (available 24/7 for emergencies)\n  - Location: {self.emergency_resources['counseling_location']}\n  - Hours: Monday-Friday, 8AM-5PM with drop-in hours 10AM-4PM\n  - Website: {self.emergency_resources['website']}\n\n• Crisis Text Line: Text HOME to {self.emergency_resources['crisis_text']}\n\n• Student Peer Support Group: Meets Tuesdays and Thursdays at 6PM in the Wellness Center\n\nIs there a specific resource you'd like more information about?",
            "faq": "That's a thoughtful question that many students ask. Let me share some insights with you based on current understanding in the field of mental health.",
            "unknown": "I appreciate you reaching out. I want to make sure I understand what you're looking for so I can provide the most helpful support. Could you tell me a bit more about what you're experiencing or what kind of assistance you're seeking today?"
        }
        
        # Select appropriate templates based on model
        templates = llama2_templates if model_name == "llama2" else llama3_templates
        
        # Format template if needed
        if intent == "slot_selection" and user_data.get("slot"):
            content = templates.get(intent, templates["unknown"]).format(slot=user_data.get("slot"))
        else:
            content = templates.get(intent, templates["unknown"])
        
        return content
    
    def test_with_ollama(self, user_input, model_name, user_data=None):
        """
        Test with actual Ollama API
        
        Args:
            user_input (str): User's message
            model_name (str): Model name (llama2 or llama3)
            user_data (dict, optional): User session data
            
        Returns:
            dict: Response data
        """
        if user_data is None:
            user_data = {}
            
        # Detect intent using the enhanced intent recognition
        intent = recognize_intent(user_input, user_data, self.available_slots)
        logger.info(f"Detected intent: {intent}")
        
        # Select the appropriate model client
        model_client = self.llama2_client if model_name == "llama2" else self.llama3_client
        
        try:
            # Handle slot selection intent
            if intent == "slot_selection":
                # Extract time slot and update user data
                slot = extract_time_slot(user_input, self.available_slots)
                if slot:
                    user_data["slot"] = slot.capitalize()
                    logger.info(f"Slot selected: {slot}")
            
            # Generate response using the enhanced Ollama integration
            start_time = time.time()
            response_data = model_client.generate_response(
                user_input, 
                [],  # Empty conversation history for testing
                user_data,
                intent
            )
            elapsed_time = time.time() - start_time
            
            # Make sure the intent is preserved
            if "intent" not in response_data or not response_data["intent"]:
                response_data["intent"] = intent
            
            # Ensure time is recorded
            response_data["time"] = elapsed_time
            
            return response_data
                
        except Exception as e:
            logger.error(f"Error with {model_name}: {str(e)}")
            # Fall back to simulation if API fails
            return self.simulate_response(user_input, model_name, user_data)
    
    def evaluate_response(self, user_input, response_data, model_name):
        """
        Evaluate response quality using the enhanced response evaluator
        
        Args:
            user_input (str): User's message
            response_data (dict): Response data
            model_name (str): Model name (llama2 or llama3)
            
        Returns:
            dict: Evaluation metrics
        """
        content = response_data["content"]
        intent = response_data["intent"]
        template_used = response_data.get("template_used", False)
        
        # Use the enhanced response evaluator
        metrics = self.response_evaluator.evaluate_response(
            user_input, content, intent, template_used
        )
        
        # Return the metrics
        return metrics
    
    def run_tests(self):
        """Run tests against both models and track intent classifications separately"""
        logger.info(f"Running {len(self.test_cases)} test cases on both models...")
        
        llama2_results = []
        llama3_results = []
        
        for i, test_case in enumerate(self.test_cases):
            user_input = test_case["input"]
            expected_intent = test_case.get("expected_intent", "unknown")
            expected_keywords = test_case.get("expected_keywords", [])
            test_category = test_case.get("category", "standard")
            
            logger.info(f"Running test {i+1}/{len(self.test_cases)}: {user_input}")
            
            # Create user data for this test
            user_data = {
                'stress': 3,  # Default stress level
                'slot': None,  # No appointment by default
                'ai_enhanced': True,  # Use enhanced AI by default
            }
            
            # Process with Llama2
            if self.ollama_available:
                try:
                    llama2_response = self.test_with_ollama(user_input, "llama2", user_data.copy())
                except Exception as e:
                    logger.error(f"Error with Llama2 API: {str(e)}")
                    llama2_response = self.simulate_response(user_input, "llama2", user_data.copy())
            else:
                llama2_response = self.simulate_response(user_input, "llama2", user_data.copy())
            
            # Process with Llama3
            if self.ollama_available:
                try:
                    llama3_response = self.test_with_ollama(user_input, "llama3", user_data.copy())
                except Exception as e:
                    logger.error(f"Error with Llama3 API: {str(e)}")
                    llama3_response = self.simulate_response(user_input, "llama3", user_data.copy())
            else:
                llama3_response = self.simulate_response(user_input, "llama3", user_data.copy())
            
            # Evaluate responses
            llama2_metrics = self.evaluate_response(user_input, llama2_response, "llama2")
            llama3_metrics = self.evaluate_response(user_input, llama3_response, "llama3")
            
            # Check for keyword matches
            llama2_keyword_match_rate = 0
            llama3_keyword_match_rate = 0
            
            if expected_keywords:
                llama2_content = llama2_response["content"].lower()
                llama3_content = llama3_response["content"].lower()
                
                llama2_matches = sum(1 for kw in expected_keywords if kw.lower() in llama2_content)
                llama3_matches = sum(1 for kw in expected_keywords if kw.lower() in llama3_content)
                
                llama2_keyword_match_rate = llama2_matches / len(expected_keywords)
                llama3_keyword_match_rate = llama3_matches / len(expected_keywords)
            
            # Ensure intent is set from responses - IMPORTANT: Use model-specific intents
            llama2_intent = llama2_response.get("intent", "unknown")
            llama3_intent = llama3_response.get("intent", "unknown")
            
            # Build result records with proper expected_intent and actual_intent
            llama2_result = {
                "test_id": i + 1,
                "input": user_input,
                "response": llama2_response["content"],
                "expected_intent": expected_intent,
                "actual_intent": llama2_intent,  # Use Llama2's actual intent
                "intent_match": expected_intent == llama2_intent,
                "expected_keywords": expected_keywords,
                "keyword_match_rate": llama2_keyword_match_rate,
                "response_length": len(llama2_response["content"].split()),
                "response_time": llama2_response.get("time", 0),
                "quality_score": llama2_metrics["overall_score"],
                "basic_quality_score": llama2_metrics["basic_quality_score"],
                "empathy_score": llama2_metrics["empathy_score"],
                "personalization": llama2_metrics["personalization"],
                "complexity": llama2_metrics["complexity"],
                "creativity": llama2_metrics["creativity"],
                "therapeutic_value": llama2_metrics["therapeutic_value"],
                "template_used": llama2_response.get("template_used", False),
                "category": test_category,
                "success": True
            }
            
            llama3_result = {
                "test_id": i + 1,
                "input": user_input,
                "response": llama3_response["content"],
                "expected_intent": expected_intent,
                "actual_intent": llama3_intent,  # Use Llama3's actual intent
                "intent_match": expected_intent == llama3_intent,
                "expected_keywords": expected_keywords,
                "keyword_match_rate": llama3_keyword_match_rate,
                "response_length": len(llama3_response["content"].split()),
                "response_time": llama3_response.get("time", 0),
                "quality_score": llama3_metrics["overall_score"],
                "basic_quality_score": llama3_metrics["basic_quality_score"],
                "empathy_score": llama3_metrics["empathy_score"],
                "personalization": llama3_metrics["personalization"],
                "complexity": llama3_metrics["complexity"],
                "creativity": llama3_metrics["creativity"],
                "therapeutic_value": llama3_metrics["therapeutic_value"],
                "template_used": llama3_response.get("template_used", False),
                "category": test_category,
                "success": True
            }
            
            llama2_results.append(llama2_result)
            llama3_results.append(llama3_result)
            
            # Print some basic info for progress tracking
            logger.info(f"  Llama2: Intent {llama2_intent}, Quality {llama2_metrics['overall_score']:.2f}")
            logger.info(f"  Llama3: Intent {llama3_intent}, Quality {llama3_metrics['overall_score']:.2f}")
        
        # Store the results for report generation
        self.llama2_results = llama2_results
        self.llama3_results = llama3_results
        
        return llama2_results, llama3_results
    
    def calculate_metrics(self, llama2_results, llama3_results):
        """Calculate aggregate metrics for both models"""
        llama2_metrics = self._calculate_model_metrics(llama2_results, "llama2")
        llama3_metrics = self._calculate_model_metrics(llama3_results, "llama3")
        
        # Calculate category-specific metrics
        llama2_metrics["category_metrics"] = self._calculate_category_metrics(llama2_results)
        llama3_metrics["category_metrics"] = self._calculate_category_metrics(llama3_results)
        
        logger.info(f"Llama2 metrics contains detailed_intent_metrics: {'detailed_intent_metrics' in llama2_metrics}")
        logger.info(f"Llama3 metrics contains detailed_intent_metrics: {'detailed_intent_metrics' in llama3_metrics}")
        
        return llama2_metrics, llama3_metrics
    
    def _calculate_model_metrics(self, results, model_name):
        """Calculate metrics for a specific model with enhanced intent classification metrics"""
        total_tests = len(results)
        
        # Basic metrics
        metrics = {
            "model_name": model_name,
            "total_tests": total_tests,
            "success_rate": sum(1 for r in results if r.get("success", False)) / total_tests,
        }
        
        # Intent recognition metrics using confusion matrix
        intent_tests = [r for r in results if "expected_intent" in r and "actual_intent" in r]
        
        logger.info(f"Found {len(intent_tests)} tests with both expected_intent and actual_intent set for {model_name}")
        
        # Process intent metrics only if we have valid intent test cases
        if intent_tests:
            # Find all unique intents for confusion matrix
            all_intents = set()
            for r in intent_tests:
                all_intents.add(r.get("expected_intent", "unknown"))
                all_intents.add(r.get("actual_intent", "unknown"))
            all_intents = sorted(list(all_intents))
            
            logger.info(f"Found {len(all_intents)} unique intents for {model_name}: {all_intents}")
            
            # Initialize confusion matrix
            confusion = {expected: {actual: 0 for actual in all_intents} for expected in all_intents}
            
            # Fill confusion matrix
            for r in intent_tests:
                expected = r.get("expected_intent", "unknown")
                actual = r.get("actual_intent", "unknown")
                if expected in confusion and actual in confusion[expected]:
                    confusion[expected][actual] = confusion[expected].get(actual, 0) + 1
            
            # Log the confusion matrix for debugging
            logger.info(f"{model_name} confusion matrix:")
            for expected in confusion:
                logger.info(f"  {expected}: {confusion[expected]}")
            
            # Calculate overall accuracy
            total_correct = sum(confusion[intent].get(intent, 0) for intent in all_intents if intent in confusion)
            total_cases = sum(sum(confusion[expected].values()) for expected in confusion)
            metrics["intent_accuracy"] = total_correct / total_cases if total_cases > 0 else 0
            
            # Calculate precision, recall, and F1 score for each intent
            intent_metrics = {}
            for intent in all_intents:
                # True positives: correctly identified this intent
                tp = confusion.get(intent, {}).get(intent, 0)
                
                # False positives: incorrectly identified as this intent
                fp = sum(confusion.get(other, {}).get(intent, 0) for other in all_intents if other != intent)
                
                # False negatives: should have been this intent but identified as something else
                fn = sum(confusion.get(intent, {}).get(other, 0) for other in all_intents if other != intent)
                
                # Calculate metrics (avoid division by zero)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                intent_metrics[intent] = {
                    "precision": precision, 
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "support": tp + fn  # Total examples of this intent
                }
            
            # Calculate macro-averaged metrics (average across all intents)
            metrics["intent_precision"] = sum(m["precision"] for m in intent_metrics.values()) / len(intent_metrics)
            metrics["intent_recall"] = sum(m["recall"] for m in intent_metrics.values()) / len(intent_metrics)
            metrics["intent_f1"] = sum(m["f1"] for m in intent_metrics.values()) / len(intent_metrics)
            
            # Calculate weighted-averaged metrics (weighted by support)
            total_support = sum(m["support"] for m in intent_metrics.values())
            if total_support > 0:
                metrics["intent_precision_weighted"] = sum(m["precision"] * m["support"] for m in intent_metrics.values()) / total_support
                metrics["intent_recall_weighted"] = sum(m["recall"] * m["support"] for m in intent_metrics.values()) / total_support
                metrics["intent_f1_weighted"] = sum(m["f1"] * m["support"] for m in intent_metrics.values()) / total_support
            else:
                metrics["intent_precision_weighted"] = 0
                metrics["intent_recall_weighted"] = 0
                metrics["intent_f1_weighted"] = 0
            
            # Store the detailed intent metrics and confusion matrix
            metrics["detailed_intent_metrics"] = intent_metrics
            metrics["confusion_matrix"] = confusion
            
        else:
            logger.warning(f"No valid intent test cases found for {model_name}. Using default values.")
            
            # Initialize with default values
            metrics["intent_accuracy"] = 0.0
            metrics["intent_precision"] = 0.0
            metrics["intent_recall"] = 0.0
            metrics["intent_f1"] = 0.0
            metrics["intent_precision_weighted"] = 0.0
            metrics["intent_recall_weighted"] = 0.0
            metrics["intent_f1_weighted"] = 0.0
            metrics["detailed_intent_metrics"] = {}
        
        # Template usage
        metrics["template_usage_rate"] = sum(1 for r in results if r.get("template_used", False)) / total_tests
        
        # Keyword match rate
        keyword_test_cases = [r for r in results if r.get("expected_keywords") and len(r.get("expected_keywords", [])) > 0]
        if keyword_test_cases:
            metrics["avg_keyword_match"] = sum(r.get("keyword_match_rate", 0) for r in keyword_test_cases) / len(keyword_test_cases)
        else:
            metrics["avg_keyword_match"] = 0.0
        
        # Quality metrics
        metrics["avg_quality_score"] = sum(r.get("quality_score", 0) for r in results) / total_tests
        metrics["avg_basic_quality_score"] = sum(r.get("basic_quality_score", 0) for r in results) / total_tests
        metrics["avg_empathy_score"] = sum(r.get("empathy_score", 0) for r in results) / total_tests
        metrics["avg_personalization"] = sum(r.get("personalization", 0) for r in results) / total_tests
        metrics["avg_complexity"] = sum(r.get("complexity", 0) for r in results) / total_tests
        metrics["avg_creativity"] = sum(r.get("creativity", 0) for r in results) / total_tests
        metrics["avg_therapeutic_value"] = sum(r.get("therapeutic_value", 0) for r in results) / total_tests
        
        # Response length and time
        metrics["avg_response_length"] = sum(r.get("response_length", 0) for r in results) / total_tests
        metrics["avg_response_time"] = sum(r.get("response_time", 0) for r in results) / total_tests
        
        return metrics
    
    def _calculate_category_metrics(self, results):
        """Calculate metrics by test category"""
        # Group results by category
        categories = set(r.get("category", "standard") for r in results)
        category_metrics = {}
        
        for category in categories:
            category_results = [r for r in results if r.get("category") == category]
            total = len(category_results)
            
            if total == 0:
                continue
            
            # Calculate average metrics for this category
            category_metrics[category] = {
                "count": total,
                "avg_quality": sum(r.get("quality_score", 0) for r in category_results) / total,
                "avg_empathy": sum(r.get("empathy_score", 0) for r in category_results) / total,
                "avg_personalization": sum(r.get("personalization", 0) for r in category_results) / total,
                "avg_complexity": sum(r.get("complexity", 0) for r in category_results) / total,
                "avg_creativity": sum(r.get("creativity", 0) for r in category_results) / total,
                "avg_therapeutic": sum(r.get("therapeutic_value", 0) for r in category_results) / total,
                "intent_accuracy": sum(1 for r in category_results if r.get("intent_match", False)) / total,
                "template_usage_rate": sum(1 for r in category_results if r.get("template_used", False)) / total
            }
        
        return category_metrics
    
    def save_results(self, llama2_results, llama3_results, llama2_metrics, llama3_metrics):
        """Save results to disk for analysis"""
        os.makedirs("results", exist_ok=True)
        
        # Save individual results
        with open("results/llama_comparison_llama2.json", "w") as f:
            json.dump(llama2_results, f, indent=2)
        
        with open("results/llama_comparison_llama3.json", "w") as f:
            json.dump(llama3_results, f, indent=2)
        
        # Save metrics
        metrics = {
            "llama2": llama2_metrics,
            "llama3": llama3_metrics,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("results/llama_comparison_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save app-friendly metrics format - this is what app.py will read
        app_metrics = {
            "llama2": {
                "avg_quality_score": llama2_metrics["avg_quality_score"],
                "avg_empathy_score": llama2_metrics["avg_empathy_score"],
                "avg_personalization": llama2_metrics["avg_personalization"],
                "template_usage_rate": llama2_metrics["template_usage_rate"],
                "avg_response_time": llama2_metrics["avg_response_time"],
                "intent_accuracy": llama2_metrics["intent_accuracy"],
                "category_metrics": llama2_metrics["category_metrics"]
            },
            "llama3": {
                "avg_quality_score": llama3_metrics["avg_quality_score"],
                "avg_empathy_score": llama3_metrics["avg_empathy_score"],
                "avg_personalization": llama3_metrics["avg_personalization"],
                "template_usage_rate": llama3_metrics["template_usage_rate"],
                "avg_response_time": llama3_metrics["avg_response_time"],
                "intent_accuracy": llama3_metrics["intent_accuracy"],
                "category_metrics": llama3_metrics["category_metrics"]
            }
        }
        
        # This is the file that app.py will read for metrics
        with open("results/llama_comparison_app_metrics.json", "w") as f:
            json.dump(app_metrics, f, indent=2)
        
        logger.info("Results and metrics saved to results directory")
        
        return metrics
    
    def generate_visualizations(self, llama2_metrics, llama3_metrics, output_dir="results"):
        """
        Generate visualizations for model comparison
        
        Args:
            llama2_metrics (dict): Metrics for Llama2
            llama3_metrics (dict): Metrics for Llama3
            output_dir (str): Output directory
        """
        if not visualization_available:
            logger.warning("Visualization libraries not available. Skipping visualization generation.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Set the style
            sns.set(style="whitegrid")
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 12
            
            # 1. Radar Chart for Advanced Metrics
            categories = ['Response Quality', 'Empathy', 'Personalization', 
                        'Complexity', 'Creativity', 'Therapeutic Value']
            
            llama2_values = [
                llama2_metrics.get('avg_quality_score', 0) * 100,
                llama2_metrics.get('avg_empathy_score', 0) * 100,
                llama2_metrics.get('avg_personalization', 0) * 100,
                llama2_metrics.get('avg_complexity', 0) * 100,
                llama2_metrics.get('avg_creativity', 0) * 100, 
                llama2_metrics.get('avg_therapeutic_value', 0) * 100
            ]
            
            llama3_values = [
                llama3_metrics.get('avg_quality_score', 0) * 100,
                llama3_metrics.get('avg_empathy_score', 0) * 100,
                llama3_metrics.get('avg_personalization', 0) * 100,
                llama3_metrics.get('avg_complexity', 0) * 100,
                llama3_metrics.get('avg_creativity', 0) * 100,
                llama3_metrics.get('avg_therapeutic_value', 0) * 100
            ]
            
            # Create radar chart
            self._create_radar_chart(categories, llama2_values, llama3_values, output_dir)
            
            # Create bar chart
            self._create_bar_chart(llama2_metrics, llama3_metrics, output_dir)
            
            # Create response time chart
            self._create_time_comparison_chart(llama2_metrics, llama3_metrics, output_dir)
            
            # Create template usage chart
            self._create_template_usage_chart(llama2_metrics, llama3_metrics, output_dir)
            
            # Create category comparison chart
            self._create_category_comparison_chart(llama2_metrics, llama3_metrics, output_dir)
            
            # Add the empathy comparison chart
            self._create_empathy_comparison_chart(llama2_metrics, llama3_metrics, output_dir)

            logger.info(f"Visualizations generated and saved to {output_dir}")
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def _create_radar_chart(self, categories, llama2_values, llama3_values, output_dir):
        """Create and save radar chart"""
        # Convert to numpy arrays
        values = np.array([llama2_values, llama3_values])
        
        # Create angles for each category
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add the first dataset (Llama2)
        llama2_values_closed = llama2_values + [llama2_values[0]]  # Close the loop
        ax.plot(angles, llama2_values_closed, 'o-', linewidth=2, label='Llama2', color='#4285f4')
        ax.fill(angles, llama2_values_closed, alpha=0.25, color='#4285f4')
        
        # Add the second dataset (Llama3)
        llama3_values_closed = llama3_values + [llama3_values[0]]  # Close the loop
        ax.plot(angles, llama3_values_closed, 'o-', linewidth=2, label='Llama3', color='#34a853')
        ax.fill(angles, llama3_values_closed, alpha=0.25, color='#34a853')
        
        # Set ticks and labels
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 100)
        ax.grid(True)
        
        # Add title and legend
        ax.set_title('Model Performance Comparison', fontsize=15)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save the plot
        plt.savefig(f"{output_dir}/radar_chart.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_bar_chart(self, llama2_metrics, llama3_metrics, output_dir):
        """Create and save bar chart for key metrics"""
        metrics = ['Response Quality', 'Empathy', 'Personalization', 'Intent Accuracy']
        
        # Values for the bar chart
        llama2_bar_values = [
            llama2_metrics.get('avg_quality_score', 0) * 100,
            llama2_metrics.get('avg_empathy_score', 0) * 100,
            llama2_metrics.get('avg_personalization', 0) * 100,
            llama2_metrics.get('intent_accuracy', 0) * 100
        ]
        
        llama3_bar_values = [
            llama3_metrics.get('avg_quality_score', 0) * 100,
            llama3_metrics.get('avg_empathy_score', 0) * 100,
            llama3_metrics.get('avg_personalization', 0) * 100,
            llama3_metrics.get('intent_accuracy', 0) * 100
        ]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(metrics))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, llama2_bar_values, width, label='Llama2', color='#4285f4')
        ax.bar(x + width/2, llama3_bar_values, width, label='Llama3', color='#34a853')
        
        # Add labels and title
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score (%)')
        ax.set_title('Key Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Add values on top of bars
        for i, v in enumerate(llama2_bar_values):
            ax.text(i - width/2, v + 1, f"{v:.1f}%", ha='center')
        
        for i, v in enumerate(llama3_bar_values):
            ax.text(i + width/2, v + 1, f"{v:.1f}%", ha='center')
        
        # Save the plot
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_time_comparison_chart(self, llama2_metrics, llama3_metrics, output_dir):
        """Create and save response time comparison chart"""
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Get average response times
        llama2_time = llama2_metrics.get("avg_response_time", 0)
        llama3_time = llama3_metrics.get("avg_response_time", 0)
        
        # Create bar chart
        models = ['Llama2', 'Llama3']
        times = [llama2_time, llama3_time]
        
        bars = plt.bar(models, times, color=['#4285f4', '#34a853'])
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=12)
        
        plt.title('Average Response Time Comparison', fontsize=14)
        plt.ylabel('Response Time (seconds)', fontsize=12)
        plt.ylim(0, max(times) * 1.2)  # Add headroom for labels
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/response_time_chart.png", dpi=300)
        plt.close()
    
    def _create_template_usage_chart(self, llama2_metrics, llama3_metrics, output_dir):
        """Create and save template usage comparison chart"""
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Get template usage rates
        llama2_template = llama2_metrics.get("template_usage_rate", 0) * 100
        llama3_template = llama3_metrics.get("template_usage_rate", 0) * 100
        
        # Create bar chart
        models = ['Llama2', 'Llama3']
        template_rates = [llama2_template, llama3_template]
        
        bars = plt.bar(models, template_rates, color=['#4285f4', '#34a853'])
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=12)
        
        plt.title('Template Usage Rate Comparison', fontsize=14)
        plt.ylabel('Template Usage (%)', fontsize=12)
        plt.ylim(0, max(template_rates) * 1.2)  # Add headroom for labels
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/template_usage_chart.png", dpi=300)
        plt.close()
    
    def _create_category_comparison_chart(self, llama2_metrics, llama3_metrics, output_dir):
        """Create and save category-specific performance charts"""
        # Get categories from both models
        llama2_categories = llama2_metrics.get("category_metrics", {})
        llama3_categories = llama3_metrics.get("category_metrics", {})
        
        # Combine categories from both models
        all_categories = sorted(set(list(llama2_categories.keys()) + list(llama3_categories.keys())))
        
        if not all_categories:
            logger.warning("No category metrics available. Skipping category chart.")
            return
        
        # Create figure for category comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(all_categories))
        width = 0.35
        
        # Get quality scores for each category
        llama2_quality = [llama2_categories.get(cat, {}).get("avg_quality", 0) * 100 for cat in all_categories]
        llama3_quality = [llama3_categories.get(cat, {}).get("avg_quality", 0) * 100 for cat in all_categories]
        
        # Create bars
        ax.bar(x - width/2, llama2_quality, width, label='Llama2', color='#4285f4')
        ax.bar(x + width/2, llama3_quality, width, label='Llama3', color='#34a853')
        
        # Add labels and title
        ax.set_xlabel('Categories')
        ax.set_ylabel('Quality Score (%)')
        ax.set_title('Response Quality by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories)
        ax.legend()
        
        # Add values on top of bars
        for i, v in enumerate(llama2_quality):
            ax.text(i - width/2, v + 1, f"{v:.1f}%", ha='center')
            
        for i, v in enumerate(llama3_quality):
            ax.text(i + width/2, v + 1, f"{v:.1f}%", ha='center')
        
        # Save the plot
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/category_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _create_empathy_comparison_chart(self, llama2_metrics, llama3_metrics, output_dir):
        """Create and save empathy comparison chart by category"""
        # Get categories from both models
        llama2_categories = llama2_metrics.get("category_metrics", {})
        llama3_categories = llama3_metrics.get("category_metrics", {})
        
        # Combine categories from both models
        all_categories = sorted(set(list(llama2_categories.keys()) + list(llama3_categories.keys())))
        
        if not all_categories:
            logger.warning("No category metrics available. Skipping empathy chart.")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(all_categories))
        width = 0.35
        
        # Get empathy scores for each category
        llama2_empathy = [llama2_categories.get(cat, {}).get("avg_empathy", 0) * 100 for cat in all_categories]
        llama3_empathy = [llama3_categories.get(cat, {}).get("avg_empathy", 0) * 100 for cat in all_categories]
        
        # Create bars
        ax.bar(x - width/2, llama2_empathy, width, label='Llama2', color='#4285f4')
        ax.bar(x + width/2, llama3_empathy, width, label='Llama3', color='#34a853')
        
        # Add labels and title
        ax.set_xlabel('Categories')
        ax.set_ylabel('Empathy Score (%)')
        ax.set_title('Empathy Score by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories)
        ax.legend()
        
        # Add values on top of bars
        for i, v in enumerate(llama2_empathy):
            ax.text(i - width/2, v + 1, f"{v:.1f}%", ha='center')
            
        for i, v in enumerate(llama3_empathy):
            ax.text(i + width/2, v + 1, f"{v:.1f}%", ha='center')
        
        # Save the plot
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/empathy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    def generate_report(self, llama2_metrics, llama3_metrics, filename="results/llama_comparison_report.html"):
        """Generate an enhanced HTML report comparing the models with interactive charts"""
        if not visualization_available:
            logger.warning("Matplotlib not available, skipping report generation")
            return
        
        # Generate visualizations (still needed for non-interactive images)
        self.generate_visualizations(llama2_metrics, llama3_metrics, "results")
        
        # Create HTML report content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Llama2 vs Llama3 Comparison Report</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #4285f4; color: white; padding: 15px; margin-bottom: 20px; }}
                .metrics-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
                .model-metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1; min-width: 300px; }}
                .model-llama2 {{ border-left: 5px solid #4285f4; }}
                .model-llama3 {{ border-left: 5px solid #34a853; }}
                .metric-row {{ display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 8px 0; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ font-family: monospace; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .comparison-table {{ width: 100%; border-collapse: collapse; margin: 25px 0; }}
                .comparison-table th {{ background-color: #f2f2f2; padding: 12px; text-align: left; }}
                .comparison-table td {{ border-bottom: 1px solid #ddd; padding: 12px; }}
                .comparison-table tr:hover {{ background-color: #f9f9f9; }}
                .chart-container {{ width: 100%; display: flex; flex-wrap: wrap; gap: 20px; margin: 30px 0; }}
                .chart {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1; min-width: 300px; height: 400px; }}
                .winner {{ font-weight: bold; background-color: #d4edda; }}
                h2, h3 {{ color: #333; margin-top: 30px; }}
                .comparison-section {{ margin-bottom: 40px; }}
                .test-case {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
                .test-input {{ font-weight: bold; margin-bottom: 10px; }}
                .response-container {{ display: flex; gap: 15px; flex-wrap: wrap; }}
                .model-response {{ flex: 1; min-width: 300px; background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }}
                .response-header {{ font-weight: bold; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #eee; }}
                .response-metrics {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
                .response-metric {{ background-color: #f1f3f4; padding: 5px 10px; border-radius: 15px; font-size: 0.9em; }}
                .category-section {{ margin-top: 40px; margin-bottom: 40px; }}
                .template-used {{ background-color: #fff8e1; }}
                .enhanced-metrics {{ background-color: #e3f2fd; padding: 5px 10px; border-radius: 15px; font-size: 0.9em; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Enhanced Llama2 vs Llama3 Comparison Report</h1>
                <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                <p>Total test cases: {len(self.test_cases)}</p>
            </div>
            
            <h2>Performance Metrics Comparison</h2>
            <div class="metrics-container">
                <div class="model-metrics model-llama2">
                    <h3>Llama2 Metrics</h3>
                    <div class="metric-row">
                        <span class="metric-name">Success Rate:</span>
                        <span class="metric-value {
                        'success' if llama2_metrics['success_rate'] > 0.8 else 'failure'
                        }">{llama2_metrics['success_rate'] * 100:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Intent Accuracy:</span>
                        <span class="metric-value {
                        'success' if llama2_metrics['intent_accuracy'] > 0.8 else 'failure'
                        }">{llama2_metrics['intent_accuracy'] * 100:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Keyword Match Rate:</span>
                        <span class="metric-value {
                        'success' if llama2_metrics.get('avg_keyword_match', 0) > 0.7 else 'failure'
                        }">{llama2_metrics.get('avg_keyword_match', 0) * 100:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Response Quality:</span>
                        <span class="metric-value {
                        'success' if llama2_metrics['avg_quality_score'] > 0.7 else 'failure'
                        }">{llama2_metrics['avg_quality_score'] * 100:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Avg Response Length:</span>
                        <span class="metric-value">{llama2_metrics['avg_response_length']:.1f} words</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Avg Response Time:</span>
                        <span class="metric-value">{llama2_metrics['avg_response_time']:.2f} sec</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Template Usage Rate:</span>
                        <span class="metric-value">{llama2_metrics['template_usage_rate'] * 100:.1f}%</span>
                    </div>
                </div>
                <div class="model-metrics model-llama3">
                    <h3>Llama3 Metrics</h3>
                    <div class="metric-row">
                        <span class="metric-name">Success Rate:</span>
                        <span class="metric-value {
                        'success' if llama3_metrics['success_rate'] > 0.8 else 'failure'
                        }">{llama3_metrics['success_rate'] * 100:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Intent Accuracy:</span>
                        <span class="metric-value {
                        'success' if llama3_metrics['intent_accuracy'] > 0.8 else 'failure'
                        }">{llama3_metrics['intent_accuracy'] * 100:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Keyword Match Rate:</span>
                        <span class="metric-value {
                        'success' if llama3_metrics.get('avg_keyword_match', 0) > 0.7 else 'failure'
                        }">{llama3_metrics.get('avg_keyword_match', 0) * 100:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Response Quality:</span>
                        <span class="metric-value {
                        'success' if llama3_metrics['avg_quality_score'] > 0.7 else 'failure'
                        }">{llama3_metrics['avg_quality_score'] * 100:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Avg Response Length:</span>
                        <span class="metric-value">{llama3_metrics['avg_response_length']:.1f} words</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Avg Response Time:</span>
                        <span class="metric-value">{llama3_metrics['avg_response_time']:.2f} sec</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">Template Usage Rate:</span>
                        <span class="metric-value">{llama3_metrics['template_usage_rate'] * 100:.1f}%</span>
                    </div>
                </div>
            </div>
            
            <h2>Advanced Metrics Comparison</h2>
            <div class="chart-container">
                <div class="chart">
                    <canvas id="radarChart"></canvas>
                </div>
                <div class="chart">
                    <canvas id="barChart"></canvas>
                </div>
            </div>
            
            <script>
                // Radar Chart for Advanced Metrics
                var radarCtx = document.getElementById('radarChart').getContext('2d');
                var radarChart = new Chart(radarCtx, {{
                    type: 'radar',
                    data: {{
                        labels: ['Response Quality', 'Empathy', 'Personalization', 'Complexity', 'Creativity', 'Therapeutic Value'],
                        datasets: [
                            {{
                                label: 'Llama2',
                                data: [{llama2_metrics['avg_quality_score'] * 100}, 
                                    {llama2_metrics['avg_empathy_score'] * 100}, 
                                    {llama2_metrics['avg_personalization'] * 100}, 
                                    {llama2_metrics['avg_complexity'] * 100}, 
                                    {llama2_metrics['avg_creativity'] * 100}, 
                                    {llama2_metrics['avg_therapeutic_value'] * 100}],
                                backgroundColor: 'rgba(66, 133, 244, 0.2)',
                                borderColor: 'rgb(66, 133, 244)',
                                pointBackgroundColor: 'rgb(66, 133, 244)',
                            }},
                            {{
                                label: 'Llama3',
                                data: [{llama3_metrics['avg_quality_score'] * 100}, 
                                    {llama3_metrics['avg_empathy_score'] * 100}, 
                                    {llama3_metrics['avg_personalization'] * 100}, 
                                    {llama3_metrics['avg_complexity'] * 100}, 
                                    {llama3_metrics['avg_creativity'] * 100}, 
                                    {llama3_metrics['avg_therapeutic_value'] * 100}],
                                backgroundColor: 'rgba(52, 168, 83, 0.2)',
                                borderColor: 'rgb(52, 168, 83)',
                                pointBackgroundColor: 'rgb(52, 168, 83)',
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            r: {{
                                angleLines: {{
                                    display: true
                                }},
                                suggestedMin: 0,
                                suggestedMax: 100
                            }}
                        }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Advanced Response Metrics Comparison'
                            }}
                        }}
                    }}
                }});
                
                // Bar Chart for Template Usage
                var barCtx = document.getElementById('barChart').getContext('2d');
                var barChart = new Chart(barCtx, {{
                    type: 'bar',
                    data: {{
                        labels: ['Template Usage', 'Original Responses'],
                        datasets: [
                            {{
                                label: 'Llama2',
                                data: [
                                    {llama2_metrics['template_usage_rate'] * 100},
                                    {(1 - llama2_metrics['template_usage_rate']) * 100}
                                ],
                                backgroundColor: [
                                    'rgba(66, 133, 244, 0.6)',
                                    'rgba(66, 133, 244, 0.3)'
                                ],
                                borderColor: [
                                    'rgb(66, 133, 244)',
                                    'rgb(66, 133, 244)'
                                ],
                                borderWidth: 1
                            }},
                            {{
                                label: 'Llama3',
                                data: [
                                    {llama3_metrics['template_usage_rate'] * 100},
                                    {(1 - llama3_metrics['template_usage_rate']) * 100}
                                ],
                                backgroundColor: [
                                    'rgba(52, 168, 83, 0.6)',
                                    'rgba(52, 168, 83, 0.3)'
                                ],
                                borderColor: [
                                    'rgb(52, 168, 83)',
                                    'rgb(52, 168, 83)'
                                ],
                                borderWidth: 1
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 100,
                                title: {{
                                    display: true,
                                    text: 'Percentage'
                                }}
                            }}
                        }},
                        plugins: {{
                            title: {{
                                display: true,
                                text: 'Template Usage vs Original Responses'
                            }}
                        }}
                    }}
                }});
            </script>
            
            <h2>Direct Comparison</h2>
            <table class="comparison-table">
                <tr>
                    <th>Metric</th>
                    <th>Llama2</th>
                    <th>Llama3</th>
                    <th>Difference</th>
                    <th>Better Model</th>
                </tr>
        """
        
        # Add comparison rows
        comparison_metrics = [
            ("Success Rate", "success_rate", "higher"),
            ("Intent Accuracy", "intent_accuracy", "higher"),
            ("Intent Precision", "intent_precision", "higher"),
            ("Intent Recall", "intent_recall", "higher"),
            ("Intent F1 Score", "intent_f1", "higher"),
            ("Keyword Match Rate", "avg_keyword_match", "higher"),
            ("Response Quality", "avg_quality_score", "higher"),
            ("Empathy Score", "avg_empathy_score", "higher"),
            ("Personalization", "avg_personalization", "higher"),
            ("Complexity", "avg_complexity", "higher"),
            ("Creativity", "avg_creativity", "higher"),
            ("Therapeutic Value", "avg_therapeutic_value", "higher"),
            ("Template Usage Rate", "template_usage_rate", "lower"),  # Lower is better
            ("Response Time (sec)", "avg_response_time", "lower")     # Lower is better
        ]
        
        for metric_name, metric_key, better in comparison_metrics:
            # Skip metrics that might not be available
            if metric_key not in llama2_metrics or metric_key not in llama3_metrics:
                continue
                
            llama2_value = llama2_metrics[metric_key]
            llama3_value = llama3_metrics[metric_key]
            
            if better == "lower":
                # For metrics where lower is better
                difference = llama2_value - llama3_value
                winner = "Llama3" if llama3_value < llama2_value else "Llama2"
            else:
                # For metrics where higher is better
                difference = llama3_value - llama2_value
                winner = "Llama3" if llama3_value > llama2_value else "Llama2"
            
            # If they're equal, no winner
            if abs(difference) < 0.001:
                winner = "Equal"
            
            # Format values appropriately
            if metric_key not in ["avg_response_time", "template_usage_rate"]:
                formatted_llama2 = f"{llama2_value * 100:.1f}%"
                formatted_llama3 = f"{llama3_value * 100:.1f}%"
                formatted_diff = f"{difference * 100:+.1f}%"
            elif metric_key == "template_usage_rate":
                formatted_llama2 = f"{llama2_value * 100:.1f}%"
                formatted_llama3 = f"{llama3_value * 100:.1f}%"
                formatted_diff = f"{difference * 100:+.1f}%"
            else:
                formatted_llama2 = f"{llama2_value:.2f}"
                formatted_llama3 = f"{llama3_value:.2f}"
                formatted_diff = f"{difference:+.2f}"
            
            html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td class="{'winner' if winner == 'Llama2' else ''}">{formatted_llama2}</td>
                    <td class="{'winner' if winner == 'Llama3' else ''}">{formatted_llama3}</td>
                    <td>{formatted_diff}</td>
                    <td>{winner}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Category-Specific Performance</h2>
        """
        
        # Add category-specific metrics
        categories = set(list(llama2_metrics.get("category_metrics", {}).keys()) + 
                        list(llama3_metrics.get("category_metrics", {}).keys()))
        
        for category in categories:
            llama2_cat = llama2_metrics.get("category_metrics", {}).get(category, {})
            llama3_cat = llama3_metrics.get("category_metrics", {}).get(category, {})
            
            if not llama2_cat and not llama3_cat:
                continue
                
            count = max(llama2_cat.get("count", 0), llama3_cat.get("count", 0))
            
            html += f"""
            <div class="category-section">
                <h3>Category: {category} ({count} tests)</h3>
                <table class="comparison-table">
                    <tr>
                        <th>Metric</th>
                        <th>Llama2</th>
                        <th>Llama3</th>
                        <th>Difference</th>
                        <th>Better Model</th>
                    </tr>
            """
            
            # Add comparison rows for this category
            category_metrics = [
                ("Response Quality", "avg_quality", "higher"),
                ("Empathy Score", "avg_empathy", "higher"),
                ("Personalization", "avg_personalization", "higher"),
                ("Complexity", "avg_complexity", "higher"),
                ("Creativity", "avg_creativity", "higher"),
                ("Therapeutic Value", "avg_therapeutic", "higher"),
                ("Intent Accuracy", "intent_accuracy", "higher"),
                ("Template Usage Rate", "template_usage_rate", "lower")
            ]
            
            for metric_name, metric_key, better in category_metrics:
                if not llama2_cat or not llama3_cat:
                    continue
                    
                # Skip metrics that might not be available
                if metric_key not in llama2_cat or metric_key not in llama3_cat:
                    continue
                    
                llama2_value = llama2_cat.get(metric_key, 0)
                llama3_value = llama3_cat.get(metric_key, 0)
                
                if better == "lower":
                    # For metrics where lower is better
                    difference = llama2_value - llama3_value
                    winner = "Llama3" if llama3_value < llama2_value else "Llama2"
                else:
                    # For metrics where higher is better
                    difference = llama3_value - llama2_value
                    winner = "Llama3" if llama3_value > llama2_value else "Llama2"
                
                # If they're equal, no winner
                if abs(difference) < 0.001:
                    winner = "Equal"
                
                # Format values
                if metric_key == "template_usage_rate":
                    formatted_llama2 = f"{llama2_value * 100:.1f}%"
                    formatted_llama3 = f"{llama3_value * 100:.1f}%"
                    formatted_diff = f"{difference * 100:+.1f}%"
                else:
                    formatted_llama2 = f"{llama2_value * 100:.1f}%"
                    formatted_llama3 = f"{llama3_value * 100:.1f}%"
                    formatted_diff = f"{difference * 100:+.1f}%"
                
                html += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td class="{'winner' if winner == 'Llama2' else ''}">{formatted_llama2}</td>
                        <td class="{'winner' if winner == 'Llama3' else ''}">{formatted_llama3}</td>
                        <td>{formatted_diff}</td>
                        <td>{winner}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        html += """
            <h2>Sample Test Cases and Responses</h2>
            <p>Below are some representative test cases showing how each model responded:</p>
        """
        if not hasattr(self, 'llama2_results') or not hasattr(self, 'llama3_results'):
            html += """
            <p><em>No test case examples available.</em></p>
            """
        else:
            # Helper function to get sample cases by category
            def get_sample_by_category(category, count=1):
                matches = []
                for i, result in enumerate(self.llama2_results):
                    if result.get("category") == category:
                        matches.append(i)
                
                # Return requested number of matches or all if fewer are available
                return matches[:min(count, len(matches))]
            
            # Get samples from each category
            sample_indices = []
            all_categories = list(categories)
            
            # Default to "standard" category if available
            if "standard" in all_categories:
                sample_indices.extend(get_sample_by_category("standard", 2))
            
            # Add samples from other categories
            for category in all_categories:
                if category != "standard":
                    sample_indices.extend(get_sample_by_category(category, 1))
            
            # Add more samples if needed to get at least 3
            if len(sample_indices) < 3 and len(self.llama2_results) > 0:
                remaining = 3 - len(sample_indices)
                
                # Get indices not already included
                additional = [i for i in range(min(len(self.llama2_results), 5)) if i not in sample_indices]
                sample_indices.extend(additional[:remaining])
        
            for idx in sample_indices:
                if idx < len(self.llama2_results) and idx < len(self.llama3_results):
                    llama2_result = self.llama2_results[idx]
                    llama3_result = self.llama3_results[idx]
                    
                    # Determine CSS classes for templates
                    llama2_class = "template-used" if llama2_result.get("template_used", False) else ""
                    llama3_class = "template-used" if llama3_result.get("template_used", False) else ""
                    
                    html += f"""
                    <div class="test-case">
                        <div class="test-input">Test ID: {llama2_result['test_id']} - Category: {llama2_result.get('category', 'standard')}</div>
                        <div class="test-input">Input: "{llama2_result['input'][:100]}{'...' if len(llama2_result['input']) > 100 else ''}"</div>
                        <div class="test-details">
                            <p>Expected Intent: <strong>{llama2_result.get('expected_intent', 'N/A')}</strong></p>
                        </div>
                        <div class="response-container">
                            <div class="model-response {llama2_class}">
                                <div class="response-header">Llama2 Response {' (Template Used)' if llama2_result.get('template_used', False) else ''}</div>
                                <p>{llama2_result['response'][:200]}{'...' if len(llama2_result['response']) > 200 else ''}</p>
                                <div class="response-metrics">
                                    <span class="response-metric">Intent: {llama2_result.get('actual_intent', 'unknown')}</span>
                                    <span class="response-metric">Match: {'✓' if llama2_result.get('intent_match', False) else '✗'}</span>
                                    <span class="response-metric">Quality: {llama2_result.get('quality_score', 0)*100:.1f}%</span>
                                    <span class="response-metric">Time: {llama2_result.get('response_time', 0):.2f}s</span>
                                    <span class="enhanced-metrics">Empathy: {llama2_result.get('empathy_score', 0)*100:.1f}%</span>
                                    <span class="enhanced-metrics">Personalization: {llama2_result.get('personalization', 0)*100:.1f}%</span>
                                    <span class="enhanced-metrics">Complexity: {llama2_result.get('complexity', 0)*100:.1f}%</span>
                                </div>
                            </div>
                            <div class="model-response {llama3_class}">
                                <div class="response-header">Llama3 Response {' (Template Used)' if llama3_result.get('template_used', False) else ''}</div>
                                <p>{llama3_result['response'][:200]}{'...' if len(llama3_result['response']) > 200 else ''}</p>
                                <div class="response-metrics">
                                    <span class="response-metric">Intent: {llama3_result.get('actual_intent', 'unknown')}</span>
                                    <span class="response-metric">Match: {'✓' if llama3_result.get('intent_match', False) else '✗'}</span>
                                    <span class="response-metric">Quality: {llama3_result.get('quality_score', 0)*100:.1f}%</span>
                                    <span class="response-metric">Time: {llama3_result.get('response_time', 0):.2f}s</span>
                                    <span class="enhanced-metrics">Empathy: {llama3_result.get('empathy_score', 0)*100:.1f}%</span>
                                    <span class="enhanced-metrics">Personalization: {llama3_result.get('personalization', 0)*100:.1f}%</span>
                                    <span class="enhanced-metrics">Complexity: {llama3_result.get('complexity', 0)*100:.1f}%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
        
        # Determine overall winner
        llama2_wins = 0
        llama3_wins = 0
        for _, metric_key, better in comparison_metrics:
            if metric_key not in llama2_metrics or metric_key not in llama3_metrics:
                continue
                
            if better == "higher":
                if llama2_metrics[metric_key] > llama3_metrics[metric_key]:
                    llama2_wins += 1
                elif llama3_metrics[metric_key] > llama2_metrics[metric_key]:
                    llama3_wins += 1
            else:  # lower is better
                if llama2_metrics[metric_key] < llama3_metrics[metric_key]:
                    llama2_wins += 1
                elif llama3_metrics[metric_key] < llama2_metrics[metric_key]:
                    llama3_wins += 1
        
        winner = "Llama3" if llama3_wins > llama2_wins else "Llama2"
        if llama2_wins == llama3_wins:
            winner = "Tie"
        
        
        html += f"""
            <h2>Conclusion</h2>
            <p>Based on the metrics above, <strong>{winner}</strong> {'performs better overall' if winner != 'Tie' else 'and Llama2 perform similarly'}.</p>
            
            <ul>
                <li>Llama2 won in {llama2_wins} categories</li>
                <li>Llama3 won in {llama3_wins} categories</li>
            </ul>
            
            <p><strong>Key findings:</strong></p>
            <ul>
                <li><strong>Response Quality:</strong> {'Llama3' if llama3_metrics['avg_quality_score'] > llama2_metrics['avg_quality_score'] else 'Llama2'} produces higher overall quality responses with a score of {max(llama3_metrics['avg_quality_score'], llama2_metrics['avg_quality_score'])*100:.1f}%</li>
                <li><strong>Empathy:</strong> {'Llama3' if llama3_metrics['avg_empathy_score'] > llama2_metrics['avg_empathy_score'] else 'Llama2'} shows more empathy with a score of {max(llama3_metrics['avg_empathy_score'], llama2_metrics['avg_empathy_score'])*100:.1f}%</li>
                <li><strong>Personalization:</strong> {'Llama3' if llama3_metrics['avg_personalization'] > llama2_metrics['avg_personalization'] else 'Llama2'} provides more personalized responses with a score of {max(llama3_metrics['avg_personalization'], llama2_metrics['avg_personalization'])*100:.1f}%</li>
                <li><strong>Template Usage:</strong> {'Llama3' if llama3_metrics['template_usage_rate'] < llama2_metrics['template_usage_rate'] else 'Llama2'} relies less on templates ({min(llama3_metrics['template_usage_rate'], llama2_metrics['template_usage_rate'])*100:.1f}% vs {max(llama3_metrics['template_usage_rate'], llama2_metrics['template_usage_rate'])*100:.1f}%)</li>
                <li><strong>Response Time:</strong> {'Llama3' if llama3_metrics['avg_response_time'] < llama2_metrics['avg_response_time'] else 'Llama2'} is faster by {abs(llama3_metrics['avg_response_time'] - llama2_metrics['avg_response_time']):.2f} seconds on average</li>
                <li><strong>Complex Queries:</strong> {'Llama3' if llama3_metrics.get('category_metrics', {}).get('complex', {}).get('avg_quality', 0) > llama2_metrics.get('category_metrics', {}).get('complex', {}).get('avg_quality', 0) else 'Llama2'} handles complex queries better</li>
            </ul>
            
            <p><strong>Recommendations:</strong></p>
            <ul>
                <li>{'Use Llama3 for complex, nuanced interactions where personalization and empathy are important.' if llama3_metrics['avg_personalization'] > llama2_metrics['avg_personalization'] and llama3_metrics['avg_empathy_score'] > llama2_metrics['avg_empathy_score'] else 'Use Llama2 for more consistent, templated responses across a range of queries.'}</li>
                <li>{'Consider using Llama2 for quick, templated responses to common queries to improve response time.' if llama2_metrics['avg_response_time'] < llama3_metrics['avg_response_time'] else 'Consider using Llama3 for all responses to provide a more cohesive user experience.'}</li>
                <li>System prompts play a significant role in output quality - continue fine-tuning these to enhance performance.</li>
            </ul>
        </body>
        </html>
        """
        
        # Close the HTML body and document
        html += """
            </table>
            <h2>Conclusion</h2>
            <p>This report provides detailed metrics comparing Llama2 and Llama3 models, including specialized intent classification metrics.</p>
            </body>
            </html>
        """
        
        # Check if intent classification metrics are available before using them
        if ('detailed_intent_metrics' in llama2_metrics and 'detailed_intent_metrics' in llama3_metrics and
        llama2_metrics['detailed_intent_metrics'] and llama3_metrics['detailed_intent_metrics']):
            # Add intent classification section to the report
            html = self._add_intent_classification_to_html_report(llama2_metrics, llama3_metrics, html)
        else:
            logger.warning("Detailed intent metrics not available, skipping intent classification section in HTML report.")
        
        # Write the report to a file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Comparison report generated: {filename}")
        
        # Generate detailed intent classification report if metrics are available
        if ('detailed_intent_metrics' in llama2_metrics and 'detailed_intent_metrics' in llama3_metrics and
        llama2_metrics['detailed_intent_metrics'] and llama3_metrics['detailed_intent_metrics']):
            intent_report_path = self._generate_intent_classification_report(llama2_metrics, llama3_metrics, "results")
            logger.info(f"Intent classification report generated: {intent_report_path}")
        else:
            logger.warning("Detailed intent metrics not available, skipping intent classification report.")
        
        return filename
        
    def _measure_creativity(self, response):
        """Measure creative language use in the response"""
        response_lower = response.lower()
        
        # Patterns indicating creative language
        creative_patterns = [
            r'\bmetaphor\b', r'\blike a\b', r'\bas if\b', r'\bimagine\b',
            r'\bpicture\b', r'\bvisualize\b', r'\bsimilar to\b', r'\banalog\b',
            r'\balternative\b', r'\bperspective\b', r'\bview\b', r'\bthink of\b'
        ]
        
        # Count creative language markers
        count = sum(1 for pattern in creative_patterns if re.search(pattern, response_lower))
        
        # Check for analogy or metaphor paragraphs
        has_extended_metaphor = 0
        if len(response_lower) > 100:  # Only check longer responses
            metaphor_paragraphs = [
                p for p in response.split('\n\n') 
                if any(re.search(pattern, p.lower()) for pattern in creative_patterns[:6])
                and len(p) > 50
            ]
            has_extended_metaphor = 1 if metaphor_paragraphs else 0
        
        # Combine markers and extended metaphor (ensure we get a non-zero baseline)
        return max(0.1, min(1.0, (count / 3) * 0.7 + has_extended_metaphor * 0.3))


   

    def _generate_intent_classification_report(self, llama2_metrics, llama3_metrics, output_dir):
        """Generate a detailed report of intent classification metrics for both models"""
        llama2_intent_metrics = llama2_metrics.get("detailed_intent_metrics", {})
        llama3_intent_metrics = llama3_metrics.get("detailed_intent_metrics", {})
        
        if not llama2_intent_metrics or not llama3_intent_metrics:
            logger.warning("Detailed intent metrics not available. Skipping intent classification report.")
            return
        
        # Get all unique intents from both models
        all_intents = sorted(set(list(llama2_intent_metrics.keys()) + list(llama3_intent_metrics.keys())))
        
        # Create the report
        report = ["# Intent Classification Performance Report\n"]
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add overall metrics for both models
        report.append("## Overall Performance\n")
        report.append("| Metric | Llama2 | Llama3 | Difference |\n")
        report.append("|--------|--------|--------|------------|\n")
        
        # Add overall metrics rows
        overall_metrics = [
            ("Accuracy", "intent_accuracy"),
            ("Precision (macro)", "intent_precision"),
            ("Recall (macro)", "intent_recall"),
            ("F1 Score (macro)", "intent_f1"),
            ("Precision (weighted)", "intent_precision_weighted"),
            ("Recall (weighted)", "intent_recall_weighted"),
            ("F1 Score (weighted)", "intent_f1_weighted")
        ]
        
        for metric_name, metric_key in overall_metrics:
            if metric_key in llama2_metrics and metric_key in llama3_metrics:
                llama2_value = llama2_metrics[metric_key]
                llama3_value = llama3_metrics[metric_key]
                diff = llama3_value - llama2_value
                
                report.append(f"| {metric_name} | {llama2_value*100:.2f}% | {llama3_value*100:.2f}% | {diff*100:+.2f}% |\n")
        
        # Add per-intent metrics for each model
        report.append("\n## Per-Intent Performance\n")
        
        for intent in all_intents:
            report.append(f"### Intent: {intent}\n")
            report.append("| Metric | Llama2 | Llama3 | Difference |\n")
            report.append("|--------|--------|--------|------------|\n")
            
            llama2_metrics_for_intent = llama2_intent_metrics.get(intent, {})
            llama3_metrics_for_intent = llama3_intent_metrics.get(intent, {})
            
            # Intent metrics to display
            intent_metric_keys = [
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1 Score", "f1"),
                ("True Positives", "tp"),
                ("False Positives", "fp"),
                ("False Negatives", "fn"),
                ("Support (total examples)", "support")
            ]
            
            for name, key in intent_metric_keys:
                llama2_value = llama2_metrics_for_intent.get(key, 0)
                llama3_value = llama3_metrics_for_intent.get(key, 0)
                
                # Format appropriately based on metric type
                if key in ["precision", "recall", "f1"]:
                    # These are percentages
                    diff = llama3_value - llama2_value
                    report.append(f"| {name} | {llama2_value*100:.2f}% | {llama3_value*100:.2f}% | {diff*100:+.2f}% |\n")
                else:
                    # These are counts
                    diff = llama3_value - llama2_value
                    report.append(f"| {name} | {llama2_value} | {llama3_value} | {diff:+d} |\n")
        
        # Add explanation of metrics
        report.append("\n## Explanation of Metrics\n")
        report.append("- **Precision**: When the model predicts an intent, how often is it correct? (TP / (TP + FP))\n")
        report.append("- **Recall**: Out of all instances of this intent, how many did the model correctly identify? (TP / (TP + FN))\n")
        report.append("- **F1 Score**: Harmonic mean of precision and recall, providing a balance between the two. (2 * precision * recall / (precision + recall))\n")
        report.append("- **Macro Average**: Simple average across all intents, giving equal weight to each intent regardless of frequency.\n")
        report.append("- **Weighted Average**: Average weighted by the number of examples for each intent, giving more weight to common intents.\n")
        report.append("- **Support**: Number of examples of each intent in the test data.\n")
        
        # Write the report to a file
        report_path = f"{output_dir}/intent_classification_report.md"
        with open(report_path, "w") as f:
            f.writelines(report)
        
        logger.info(f"Intent classification report generated: {report_path}")
        return report_path
    
    def _add_intent_classification_to_html_report(self, llama2_metrics, llama3_metrics, html_content):
        """Add intent classification metrics to the HTML report"""
        llama2_intent_metrics = llama2_metrics.get("detailed_intent_metrics", {})
        llama3_intent_metrics = llama3_metrics.get("detailed_intent_metrics", {})
        
        if not llama2_intent_metrics or not llama3_intent_metrics:
            return html_content  # Return original content if metrics aren't available
        
        # Create the intent classification section
        intent_section = """
        <h2>Intent Classification Metrics</h2>
        <p>Detailed metrics for evaluating the performance of each model's intent classification.</p>
        
        <h3>Overall Intent Classification Performance</h3>
        <table class="comparison-table">
            <tr>
                <th>Metric</th>
                <th>Llama2</th>
                <th>Llama3</th>
                <th>Difference</th>
                <th>Better Model</th>
            </tr>
        """
        
        # Add overall metrics rows
        overall_metrics = [
            ("Accuracy", "intent_accuracy"),
            ("Precision (macro)", "intent_precision"),
            ("Recall (macro)", "intent_recall"),
            ("F1 Score (macro)", "intent_f1"),
            ("Precision (weighted)", "intent_precision_weighted"),
            ("Recall (weighted)", "intent_recall_weighted"),
            ("F1 Score (weighted)", "intent_f1_weighted")
        ]
        
        for metric_name, metric_key in overall_metrics:
            if metric_key in llama2_metrics and metric_key in llama3_metrics:
                llama2_value = llama2_metrics[metric_key]
                llama3_value = llama3_metrics[metric_key]
                diff = llama3_value - llama2_value
                
                # Determine winner
                winner = "Equal"
                if abs(diff) > 0.001:  # Small threshold to avoid floating point comparison issues
                    winner = "Llama3" if diff > 0 else "Llama2"
                
                intent_section += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td class="{'winner' if winner == 'Llama2' else ''}">{llama2_value*100:.2f}%</td>
                    <td class="{'winner' if winner == 'Llama3' else ''}">{llama3_value*100:.2f}%</td>
                    <td>{diff*100:+.2f}%</td>
                    <td>{winner}</td>
                </tr>
                """
        
        intent_section += """
        </table>
        
        <h3>Per-Intent Performance</h3>
        <div class="intent-tabs">
            <ul class="intent-tab-links">
        """
        
        # Get all unique intents
        all_intents = sorted(set(list(llama2_intent_metrics.keys()) + list(llama3_intent_metrics.keys())))
        
        # Add tab links for each intent
        for i, intent in enumerate(all_intents):
            intent_section += f"""
            <li><a href="#intent-{i}" class="{'active' if i == 0 else ''}">{intent}</a></li>
            """
        
        intent_section += """
            </ul>
            <div class="intent-tab-content">
        """
        
        # Add tab content for each intent
        for i, intent in enumerate(all_intents):
            llama2_metrics_for_intent = llama2_intent_metrics.get(intent, {})
            llama3_metrics_for_intent = llama3_intent_metrics.get(intent, {})
            
            intent_section += f"""
            <div id="intent-{i}" class="intent-tab-pane {' active' if i == 0 else ''}">
                <h4>Metrics for Intent: {intent}</h4>
                <table class="comparison-table">
                    <tr>
                        <th>Metric</th>
                        <th>Llama2</th>
                        <th>Llama3</th>
                        <th>Difference</th>
                        <th>Better Model</th>
                    </tr>
            """
            
            # Intent metrics to display
            intent_metric_keys = [
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1 Score", "f1")
            ]
            
            for name, key in intent_metric_keys:
                llama2_value = llama2_metrics_for_intent.get(key, 0)
                llama3_value = llama3_metrics_for_intent.get(key, 0)
                diff = llama3_value - llama2_value
                
                # Determine winner
                winner = "Equal"
                if abs(diff) > 0.001:  # Small threshold
                    winner = "Llama3" if diff > 0 else "Llama2"
                
                intent_section += f"""
                <tr>
                    <td>{name}</td>
                    <td class="{'winner' if winner == 'Llama2' else ''}">{llama2_value*100:.2f}%</td>
                    <td class="{'winner' if winner == 'Llama3' else ''}">{llama3_value*100:.2f}%</td>
                    <td>{diff*100:+.2f}%</td>
                    <td>{winner}</td>
                </tr>
                """
            
            # Add count metrics
            count_metrics = [
                ("True Positives", "tp"),
                ("False Positives", "fp"),
                ("False Negatives", "fn"),
                ("Support (total examples)", "support")
            ]
            
            for name, key in count_metrics:
                llama2_value = llama2_metrics_for_intent.get(key, 0)
                llama3_value = llama3_metrics_for_intent.get(key, 0)
                diff = llama3_value - llama2_value
                
                intent_section += f"""
                <tr>
                    <td>{name}</td>
                    <td>{llama2_value}</td>
                    <td>{llama3_value}</td>
                    <td>{diff:+d}</td>
                    <td>N/A</td>
                </tr>
                """
            
            intent_section += """
                </table>
            </div>
            """
        
        intent_section += """
            </div>
        </div>
        
        <h3>Explanation of Intent Classification Metrics</h3>
        <p><strong>Precision</strong>: When the model predicts a particular intent, how often is it correct? (True Positives / (True Positives + False Positives))</p>
        <p><strong>Recall</strong>: Out of all instances of a particular intent, how many did the model correctly identify? (True Positives / (True Positives + False Negatives))</p>
        <p><strong>F1 Score</strong>: The harmonic mean of precision and recall, providing a balance between the two metrics. (2 * Precision * Recall / (Precision + Recall))</p>
        <p><strong>True Positives</strong>: Number of times the model correctly identified this intent.</p>
        <p><strong>False Positives</strong>: Number of times the model incorrectly classified another intent as this one.</p>
        <p><strong>False Negatives</strong>: Number of times this intent was incorrectly classified as another intent.</p>
        <p><strong>Support</strong>: Total number of examples of this intent in the test set.</p>
        <p><strong>Macro Average</strong>: Simple average across all intents, giving equal weight to each intent regardless of frequency.</p>
        <p><strong>Weighted Average</strong>: Average weighted by the number of examples for each intent, giving more weight to common intents.</p>
        
        <style>
        .intent-tabs {
            margin: 20px 0;
        }
        .intent-tab-links {
            display: flex;
            flex-wrap: wrap;
            list-style-type: none;
            padding: 0;
            margin: 0;
            border-bottom: 1px solid #ddd;
        }
        .intent-tab-links li {
            margin-right: 5px;
        }
        .intent-tab-links a {
            display: block;
            padding: 10px 15px;
            text-decoration: none;
            color: #333;
            background-color: #f1f1f1;
            border-radius: 5px 5px 0 0;
        }
        .intent-tab-links a.active {
            background-color: #4285f4;
            color: white;
        }
        .intent-tab-content {
            border: 1px solid #ddd;
            border-top: none;
            padding: 15px;
        }
        .intent-tab-pane {
            display: none;
        }
        .intent-tab-pane.active {
            display: block;
        }
        </style>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Tab functionality
            const tabLinks = document.querySelectorAll('.intent-tab-links a');
            const tabPanes = document.querySelectorAll('.intent-tab-pane');
            
            tabLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    // Remove active class from all tabs
                    tabLinks.forEach(l => l.classList.remove('active'));
                    tabPanes.forEach(p => p.classList.remove('active'));
                    
                    // Add active class to current tab
                    this.classList.add('active');
                    const targetId = this.getAttribute('href');
                    document.querySelector(targetId).classList.add('active');
                });
            });
        });
        </script>
        """
        
        # Find position to insert the intent section (before the conclusion)
        conclusion_pos = html_content.find("<h2>Conclusion</h2>")
        if conclusion_pos == -1:
            # If conclusion section not found, append to the end
            return html_content + intent_section
        else:
            # Insert before conclusion
            return html_content[:conclusion_pos] + intent_section + html_content[conclusion_pos:]

def main():
    """Run the enhanced model comparison test"""
    print("=" * 80)
    print(f"ENHANCED LLAMA2 VS LLAMA3 COMPARISON - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    # Check if FAQ file exists
    faq_path = "data/Mental_Health_FAQ.csv"
    if not os.path.exists(faq_path):
        faq_path = None
        logger.warning("FAQ file not found. Will proceed without FAQ matching.")
    
    # Initialize the tester with the advanced components
    tester = EnhancedModelTester(faq_path)
    
    # Run the tests
    logger.info("\nRunning comparison tests...")
    start_time = time.time()
    llama2_results, llama3_results = tester.run_tests()
    elapsed_time = time.time() - start_time
    
    logger.info(f"\nTests completed in {elapsed_time:.2f} seconds")
    logger.info(f"Processed {len(llama2_results)} test cases for each model")
    
    # Calculate metrics
    logger.info("\nCalculating comparison metrics...")
    llama2_metrics, llama3_metrics = tester.calculate_metrics(llama2_results, llama3_results)
    
    # Save results
    logger.info("\nSaving detailed results...")
    metrics = tester.save_results(llama2_results, llama3_results, llama2_metrics, llama3_metrics)
    
    # Generate visualizations
    logger.info("\nGenerating comparison visualizations...")
    tester.generate_visualizations(llama2_metrics, llama3_metrics)
    
    # Generate report
    logger.info("\nGenerating comparison report...")
    report_file = tester.generate_report(llama2_metrics, llama3_metrics)
    
    logger.info("\nComparison completed successfully!")
    if report_file:
        logger.info(f"Report available at: {report_file}")
    print("\nComparison completed successfully!")
    if visualization_available:
        print(f"Report available at: {report_file}")
        print(f"Intent classification report available at: results/intent_classification_report.md")
    print("Detailed results saved in results directory")

    # Print quick summary
    print("\nQUICK SUMMARY:")
    print(f"  Llama2 Overall Quality: {llama2_metrics.get('avg_quality_score', 0)*100:.1f}%")
    print(f"  Llama3 Overall Quality: {llama3_metrics.get('avg_quality_score', 0)*100:.1f}%")
    print(f"  Llama2 Template Usage: {llama2_metrics.get('template_usage_rate', 0)*100:.1f}%")
    print(f"  Llama3 Template Usage: {llama3_metrics.get('template_usage_rate', 0)*100:.1f}%")
    print(f"  Llama2 Response Time: {llama2_metrics.get('avg_response_time', 0):.2f}s")
    print(f"  Llama3 Response Time: {llama3_metrics.get('avg_response_time', 0):.2f}s")
    print("\nResults saved to results directory for use by the web application")

if __name__ == "__main__":
    # Import random for simulations if Ollama isn't available
    import random
    main()
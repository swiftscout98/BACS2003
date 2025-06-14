import json
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
import numpy as np

class EnhancedResponseEvaluator:
    """Enhanced class to evaluate and improve chatbot response quality with more nuanced metrics"""
    
    def __init__(self):
        """Initialize the response evaluator"""
        self.response_metrics = {}
        self.response_improvements = {}
        
        # Load stopwords for evaluation
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback if NLTK resources aren't available
            self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                                   'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                                   'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                                   'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                                   'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                                   'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                                   'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                                   'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                                   'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                                   'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                                   'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                                   'again', 'further', 'then', 'once', 'here', 'there', 'when', 
                                   'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
                                   'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
                                   'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
                                   'can', 'will', 'just', 'don', 'should', 'now'])
        
        # Load therapeutic language patterns
        self.therapeutic_language = self._load_therapeutic_language()
    
    def _load_therapeutic_language(self):
        """Load therapeutic language patterns for evaluating responses"""
        return {
            "empathy": [
                r'understand', r'feeling', r'difficult', r'challenging', 
                r'appreciate', r'valid', r'normalize', r'acknowledge',
                r'perspective', r'experience', r'hear you', r'with you'
            ],
            "validation": [
                r'normal', r'makes sense', r'understandable', r'natural',
                r'valid', r'common', r'many people', r'not alone'
            ],
            "personalization": [
                r'you mentioned', r'you said', r'your situation',
                r'you shared', r'your experience', r'your feelings',
                r'in your case', r'for you specifically'
            ],
            "therapeutic_techniques": [
                r'mindfulness', r'breathing', r'meditation', r'cognitive',
                r'behavioral', r'self-care', r'reframe', r'approach',
                r'strategy', r'technique', r'exercise', r'practice'
            ],
            "support_language": [
                r'here for you', r'support', r'help', r'assist',
                r'together', r'work on', r'explore', r'discover',
                r'resource', r'service', r'counselor', r'professional'
            ]
        }
    
    def evaluate_response(self, user_input, response, intent=None, template_used=False):
        """
        Evaluate the quality of a response with enhanced metrics
        
        Args:
            user_input (str): User's question or message
            response (str): Chatbot's response
            intent (str, optional): Detected intent of the user
            template_used (bool): Whether a template was used for the response
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['length'] = len(response.split())
        metrics['length_score'] = min(1.0, metrics['length'] / 30) if metrics['length'] < 100 else 1.0
        
        # Response relevance (keyword matching)
        user_keywords = self._extract_keywords(user_input)
        response_keywords = self._extract_keywords(response)
        
        if user_keywords:
            matching_keywords = [k for k in user_keywords if k in response_keywords]
            metrics['keyword_match'] = len(matching_keywords) / len(user_keywords)
        else:
            metrics['keyword_match'] = 0.5  # Default if no keywords
        
        # Response structure
        metrics['has_greeting'] = 1.0 if self._has_greeting(response) else 0.0
        metrics['has_punctuation'] = 1.0 if self._has_proper_punctuation(response) else 0.0
        
        # Intent-specific checks
        metrics['intent_specific'] = self._check_intent_specific(response, intent)
        
        # Enhanced metrics
        metrics['empathy_score'] = self._measure_empathy(response)
        metrics['personalization'] = self._measure_personalization(response, user_input)
        metrics['complexity'] = self._measure_complexity(response)
        metrics['creativity'] = self._measure_creativity(response)
        metrics['therapeutic_value'] = self._measure_therapeutic_value(response)
        
        # Penalize template usage for comparing model capabilities
        metrics['template_penalty'] = 0.0 if not template_used else 0.2
        
        # Calculate traditional quality score (weighted average of basic metrics)
        basic_quality = (
            0.2 * metrics['length_score'] +
            0.3 * metrics['keyword_match'] +
            0.1 * metrics['has_greeting'] + 
            0.1 * metrics['has_punctuation'] +
            0.2 * metrics['intent_specific'] +
            0.1 * metrics['empathy_score']
        )
        
        # Calculate enhanced quality score (incorporating new metrics)
        enhanced_quality = (
            0.15 * metrics['length_score'] +
            0.15 * metrics['keyword_match'] +
            0.05 * metrics['has_greeting'] + 
            0.05 * metrics['has_punctuation'] +
            0.15 * metrics['intent_specific'] +
            0.15 * metrics['empathy_score'] +
            0.15 * metrics['personalization'] +
            0.05 * metrics['complexity'] +
            0.05 * metrics['creativity'] +
            0.10 * metrics['therapeutic_value'] -
            metrics['template_penalty']  # Penalty for template usage
        )
        
        # Store both scores
        metrics['basic_quality_score'] = basic_quality
        metrics['overall_score'] = enhanced_quality
        
        # Store metrics
        self.response_metrics[(user_input, response)] = metrics
        
        return metrics
    
    def _extract_keywords(self, text):
        """Extract important keywords from text"""
        words = word_tokenize(text.lower()) if text else []
        # Filter out stopwords, punctuation, and short words
        keywords = [w for w in words if w not in self.stop_words 
                   and w not in string.punctuation 
                   and len(w) > 3]
        return keywords
    
    def _has_greeting(self, text):
        """Check if response has a greeting"""
        greeting_patterns = [
            r'^hi\b', r'^hello\b', r'^hey\b', r'^\bi\b', r'^thanks for',
            r'welcome', r'good to see', r'how are you', r'hope you'
        ]
        return any(re.search(pattern, text.lower()) for pattern in greeting_patterns)
    
    def _has_proper_punctuation(self, text):
        """Check if response has proper punctuation"""
        sentences = re.split(r'[.!?]', text)
        # Check if most sentences end with punctuation
        if len(sentences) <= 1:
            return False
        return text.strip()[-1] in ['.', '!', '?']
    
    def _check_intent_specific(self, response, intent):
        """Check for intent-specific content"""
        if not intent:
            return 0.5  # Neutral if no intent
        
        intent_patterns = {
            "stress_assessment": [r'scale', r'rate', r'level', r'1.*5', r'stress level'],
            "booking": [r'appointment', r'schedule', r'available', r'slot', r'time'],
            "stress_tip": [r'breath', r'relax', r'calm', r'practice', r'try'],
            "emergency": [r'immediate', r'call', r'crisis', r'hotline', r'988'],
            "academic": [r'study', r'course', r'class', r'assignment', r'technique']
        }
        
        if intent in intent_patterns:
            patterns = intent_patterns[intent]
            matches = sum(1 for p in patterns if re.search(p, response.lower()))
            return min(1.0, matches / len(patterns))
        
        return 0.5  # Default value for other intents
    
    def _measure_empathy(self, response):
        """Measure empathetic language in response"""
        empathy_phrases = self.therapeutic_language['empathy']
        response_lower = response.lower()
        
        # Count matches
        count = sum(1 for phrase in empathy_phrases if re.search(phrase, response_lower))
        
        # Normalize score (0-1)
        return min(1.0, count / 4)  # 4 or more empathy markers for full score
    
    def _measure_personalization(self, response, user_input):
        """Measure how personalized the response is to the user input"""
        # Check for personalization markers
        personalization_phrases = self.therapeutic_language['personalization']
        response_lower = response.lower()
        
        # Count direct personalization markers
        marker_count = sum(1 for phrase in personalization_phrases 
                         if re.search(phrase, response_lower))
        
        # Check if response refers to specific elements from user input
        user_keywords = self._extract_keywords(user_input)
        if not user_keywords:
            return 0.5 if marker_count > 0 else 0.3
        
        # Count references to user's specific concerns
        mentioned = sum(1 for kw in user_keywords 
                       if re.search(fr'\b{re.escape(kw)}\b', response_lower))
        
        # Combine both metrics
        keyword_score = min(1.0, mentioned / max(1, len(user_keywords) * 0.5))
        return min(1.0, (marker_count * 0.5 + keyword_score * 0.5))
    
    def _measure_complexity(self, response):
        """Measure linguistic complexity of the response"""
        # Get sentences
        sentences = sent_tokenize(response)
        if not sentences:
            return 0.3
        
        # Calculate average sentence length
        avg_words = sum(len(word_tokenize(s)) for s in sentences) / len(sentences)
        
        # Calculate vocabulary diversity (unique word ratio)
        words = word_tokenize(response.lower())
        unique_ratio = len(set(words)) / max(1, len(words))
        
        # Calculate sentence structure complexity (approximation)
        complex_structures = 0
        connectors = ['however', 'although', 'therefore', 'because', 'since', 
                      'while', 'whereas', 'despite', 'though', 'furthermore']
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(connector in sentence_lower for connector in connectors):
                complex_structures += 1
        
        structure_complexity = min(1.0, complex_structures / max(1, len(sentences) * 0.5))
        
        # Combine metrics
        length_score = min(1.0, avg_words / 20)  # Normalize - 20 words is "complex enough"
        
        return min(1.0, (length_score * 0.4 + unique_ratio * 0.4 + structure_complexity * 0.2))
    
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
        
        # Combine markers and extended metaphor
        return min(1.0, (count / 3) * 0.7 + has_extended_metaphor * 0.3)
    
    def _measure_therapeutic_value(self, response):
        """Measure therapeutic value of the response"""
        response_lower = response.lower()
        
        # Check for therapeutic technique mentions
        technique_patterns = self.therapeutic_language['therapeutic_techniques']
        technique_count = sum(1 for pattern in technique_patterns 
                            if re.search(pattern, response_lower))
        
        # Check for validation language
        validation_patterns = self.therapeutic_language['validation']
        validation_count = sum(1 for pattern in validation_patterns
                             if re.search(pattern, response_lower))
        
        # Check for support language
        support_patterns = self.therapeutic_language['support_language']
        support_count = sum(1 for pattern in support_patterns
                          if re.search(pattern, response_lower))
        
        # Combine scores
        technique_score = min(1.0, technique_count / 3)
        validation_score = min(1.0, validation_count / 2)
        support_score = min(1.0, support_count / 2)
        
        return min(1.0, (technique_score * 0.5 + validation_score * 0.3 + support_score * 0.2))
    
    def improve_response(self, user_input, response, metrics, intent=None):
        """
        Suggest improvements for the response if quality is low
        
        Args:
            user_input (str): User's question or message
            response (str): Chatbot's response
            metrics (dict): Evaluation metrics
            intent (str, optional): Detected intent of the user
            
        Returns:
            str: Improved response or original response if quality is good
        """
        # If quality is good enough, return original response
        if metrics['overall_score'] >= 0.7:
            return response
        
        # Identify what needs improvement
        improvements = []
        
        # Check length
        if metrics['length_score'] < 0.5:
            improvements.append("expand_response")
        
        # Check keyword match
        if metrics['keyword_match'] < 0.3:
            improvements.append("increase_relevance")
        
        # Check greeting
        if metrics['has_greeting'] == 0:
            improvements.append("add_greeting")
        
        # Check intent-specific needs
        if metrics['intent_specific'] < 0.5:
            improvements.append("add_intent_specifics")
        
        if metrics['empathy_score'] < 0.5:
            improvements.append("add_emotional_support")
        
        if metrics['personalization'] < 0.4:
            improvements.append("add_personalization")
        
        # For critical intents, ensure specific content
        if intent in ["emergency", "stress_assessment"] and "add_intent_specifics" in improvements:
            # Emergency responses need immediate actionable information
            if intent == "emergency":
                improved = self._emergency_fallback()
                self.response_improvements[(user_input, response)] = improvements
                return improved
            
            # Stress assessment needs clear instructions
            if intent == "stress_assessment":
                improved = self._stress_assessment_fallback()
                self.response_improvements[(user_input, response)] = improvements
                return improved
        
        # For less critical improvements, just return original
        self.response_improvements[(user_input, response)] = improvements
        return response
    
    def _emergency_fallback(self):
        """Fallback response for emergencies"""
        return """I'm concerned about what you're sharing. If you're experiencing a mental health emergency, please:

1. Call the 24/7 Crisis Hotline at 988 immediately
2. Contact the University Counseling Center at 555-123-4567
3. If you're on campus, go to the Student Center, Room 302

Your wellbeing is important, and immediate help is available. Would you like me to provide additional resources?"""
    
    def _stress_assessment_fallback(self):
        """Fallback response for stress assessment"""
        return """I'd like to understand your stress level better. On a scale of 1-5, how would you rate your current stress?

1 = Minimal stress (feeling calm and in control)
2 = Mild stress (slightly worried but managing well)
3 = Moderate stress (noticeably tense, having some trouble coping)
4 = High stress (very worried, difficulty concentrating o1 sleeping)
5 = Severe stress (feeling overwhelmed, unable to function normally)

This will help me provide more personalized support."""
    
    def export_metrics(self, filename='response_metrics.json'):
        """Export all collected metrics to a JSON file"""
        export_data = {}
        
        for (user_input, response), metrics in self.response_metrics.items():
            # Create a hashable key
            key = f"{hash(user_input)}_{hash(response)}"
            
            export_data[key] = {
                'user_input': user_input,
                'response': response,
                'metrics': metrics
            }
            
            # Add improvements if available
            if (user_input, response) in self.response_improvements:
                export_data[key]['improvements'] = self.response_improvements[(user_input, response)]
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
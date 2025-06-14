import pandas as pd
import numpy as np
import time
from datetime import datetime
import nltk
import re
import json
import requests
import logging
import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Import the components 
from advanced_faq_matcher import AdvancedFAQMatcher
from enhanced_intent_recognition import recognize_intent, extract_time_slot
from enhanced_response_evaluator import EnhancedResponseEvaluator
from enhanced_ollama_integration import EnhancedOllamaAPI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask application
app = Flask(__name__)
app.secret_key = 'mental_health_support_2025'

# Initialize components
class MentalHealthChatbot:
    """Enhanced Mental Health Chatbot with Model Comparison"""
    
    def __init__(self, faq_path="data/Mental_Health_FAQ.csv"):
        """Initialize the chatbot components"""
        # Download NLTK resources if needed
        self._ensure_nltk_resources()
        
        # Load FAQ data
        try:
            self.faq_df = pd.read_csv(faq_path)
            # Convert Questions column to string explicitly
            self.faq_df["Questions"] = self.faq_df["Questions"].astype(str)
            logger.info(f"Loaded FAQ data with {len(self.faq_df)} entries")
        except Exception as e:
            logger.error(f"Error loading FAQ data: {str(e)}")
            # Create empty DataFrame if file not found
            self.faq_df = pd.DataFrame({"Questions": [], "Answers": []})
        
        # Initialize advanced components
        self.faq_matcher = AdvancedFAQMatcher(self.faq_df)
        self.response_evaluator = EnhancedResponseEvaluator()
        
        # Initialize both LLM models
        self.llama2_client = EnhancedOllamaAPI(model_name="llama2")
        self.llama3_client = EnhancedOllamaAPI(model_name="llama3")
        
        # Predefined appointment slots
        self.available_slots = ["mon 10 am", "tue 2 pm", "wed 11 am", "thu 3 pm", "fri 2 pm"]
        
        # Define emergency resources
        self.emergency_resources = {
            "crisis_number": "988",
            "university_number": "555-123-4567",
            "crisis_text": "741741",
            "counseling_location": "Student Center, Room 302",
            "website": "counseling.university.edu"
        }
        
        # Model performance metrics
        self.model_metrics = {
            "llama2": {"response_time": [], "quality_scores": []},
            "llama3": {"response_time": [], "quality_scores": []}
        }
        
        # Load predefined model metrics from comparison
        self._load_model_metrics()
        
        logger.info("Mental Health Chatbot initialized successfully")

    def _ensure_nltk_resources(self):
        """Download required NLTK resources if not available"""
        resources = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet')
        ]
        
        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, quiet=True)
    
    def _load_model_metrics(self):
        """Load predefined model metrics from comparison file if available"""
        try:
            with open("results/llama_comparison_metrics.json", "r") as f:
                comparison_data = json.load(f)
                
                # Extract key metrics for Report display
                self.comparison_metrics = {
                    "llama2": {
                        "response_quality": comparison_data["llama2"]["avg_quality_score"] * 100,
                        "empathy_score": comparison_data["llama2"]["avg_empathy_score"] * 100,
                        "personalization": comparison_data["llama2"]["avg_personalization"] * 100,
                        "template_usage": comparison_data["llama2"]["template_usage_rate"] * 100,
                        "response_time": comparison_data["llama2"]["avg_response_time"],
                        "intent_accuracy": comparison_data["llama2"]["intent_accuracy"] * 100
                    },
                    "llama3": {
                        "response_quality": comparison_data["llama3"]["avg_quality_score"] * 100,
                        "empathy_score": comparison_data["llama3"]["avg_empathy_score"] * 100,
                        "personalization": comparison_data["llama3"]["avg_personalization"] * 100,
                        "template_usage": comparison_data["llama3"]["template_usage_rate"] * 100,
                        "response_time": comparison_data["llama3"]["avg_response_time"],
                        "intent_accuracy": comparison_data["llama3"]["intent_accuracy"] * 100
                    }
                }
                
                # Load category-specific metrics for advanced analysis
                self.category_metrics = {
                    "llama2": comparison_data["llama2"]["category_metrics"],
                    "llama3": comparison_data["llama3"]["category_metrics"]
                }
                
                logger.info("Loaded predefined model metrics from comparison data")
        except Exception as e:
            logger.warning(f"Could not load model comparison metrics: {str(e)}")
            # Set default metrics if file not found
            self.comparison_metrics = {
                "llama2": {
                    "response_quality": 36.3,
                    "empathy_score": 23.0,
                    "personalization": 25.1,
                    "template_usage": 59.5,
                    "response_time": 6.12,
                    "intent_accuracy": 56.8
                },
                "llama3": {
                    "response_quality": 57.4,
                    "empathy_score": 42.6,
                    "personalization": 40.4,
                    "template_usage": 0.0,
                    "response_time": 23.57,
                    "intent_accuracy": 56.8
                }
            }
    
    def generate_response(self, user_input, conversation_history, user_data, active_model="llama3"):
        """
        Generate a response using the selected model
        
        Args:
            user_input (str): User's message
            conversation_history (list): Conversation history
            user_data (dict): User data including stress level, appointments, etc.
            active_model (str): Which model to use (llama2 or llama3)
            
        Returns:
            dict: Response data including content, intent, and performance metrics
        """
        start_time = time.time()
        
        # Detect intent
        intent = recognize_intent(user_input, user_data, self.available_slots)
        logger.info(f"Detected intent: {intent}")
        
        # Try FAQ matching first
        matched_question, answer, confidence = self.faq_matcher.find_match(
            user_input, 
            threshold=0.75 if active_model == "llama2" else 0.65
        )
        
        # Set appropriate confidence threshold based on model
        faq_confidence_threshold = 0.85 if active_model == "llama2" else 0.95
        template_used = False
        
        if answer and confidence > faq_confidence_threshold:
            # Use FAQ answer directly for high confidence matches
            content = answer
            logger.info(f"FAQ match found with confidence {confidence:.2f}")
        else:
            # Select the appropriate model client
            model_client = self.llama2_client if active_model == "llama2" else self.llama3_client
            
            # Handle specific intents
            if intent == "slot_selection":
                # Extract time slot and update user data
                slot = extract_time_slot(user_input, self.available_slots)
                if slot:
                    user_data["slot"] = slot.capitalize()
                    logger.info(f"Slot selected: {slot}")
            
            # Get response from selected model
            response = model_client.generate_response(
                user_input, 
                conversation_history[-5:], 
                user_data,
                intent
            )
            
            content = response.get("content", "Error generating response")
            template_used = response.get("template_used", False)
            
        # Calculate response time
        response_time = time.time() - start_time
        
        # Evaluate response quality
        quality_metrics = self.response_evaluator.evaluate_response(
            user_input, content, intent, template_used
        )
        
        # Update model metrics for Report
        self.model_metrics[active_model]["response_time"].append(response_time)
        self.model_metrics[active_model]["quality_scores"].append(quality_metrics["overall_score"])
        
        # Limit stored metrics to last 100 interactions
        if len(self.model_metrics[active_model]["response_time"]) > 100:
            self.model_metrics[active_model]["response_time"].pop(0)
        if len(self.model_metrics[active_model]["quality_scores"]) > 100:
            self.model_metrics[active_model]["quality_scores"].pop(0)
            
        return {
            "content": content,
            "intent": intent,
            "response_time": response_time,
            "quality_score": quality_metrics["overall_score"],
            "empathy_score": quality_metrics["empathy_score"],
            "personalization": quality_metrics["personalization"],
            "template_used": template_used
        }
    
    def update_stress_level(self, level, user_data):
        """
        Update user's stress level and return appropriate class
        
        Args:
            level (int): Stress level (1-5)
            user_data (dict): User data
            
        Returns:
            str: CSS class for the stress level indicator
        """
        try:
            level = int(level)
            if 1 <= level <= 5:
                user_data["stress"] = level
                
                # Add to mood history
                current_date = datetime.now().strftime("%Y-%m-%d")
                if "mood_history" not in user_data:
                    user_data["mood_history"] = []
                user_data["mood_history"].append((current_date, level))
                
                # Limit mood history to last 30 days
                if len(user_data["mood_history"]) > 30:
                    user_data["mood_history"].pop(0)
                
                # Return appropriate CSS class
                if level <= 2:
                    return "stress-low"
                elif level <= 4:
                    return "stress-medium"
                else:
                    return "stress-high"
        except Exception as e:
            logger.error(f"Error updating stress level: {str(e)}")
        
        return "stress-medium"  # Default
    
    def generate_mood_chart_image(self, mood_history):
        """
        Generate mood chart image from mood history
        
        Args:
            mood_history (list): List of (date, level) tuples
            
        Returns:
            str: Base64 encoded PNG image
        """
        try:
            # Create chart
            plt.figure(figsize=(10, 6))
            
            if mood_history:
                dates = [item[0] for item in mood_history]
                levels = [item[1] for item in mood_history]
                
                # Plot mood level over time
                plt.plot(dates, levels, marker='o', linestyle='-', color='#4285f4', linewidth=2, markersize=8)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.ylabel('Stress Level', fontsize=12)
                plt.xlabel('Date', fontsize=12)
                plt.title('Your Stress Level Over Time', fontsize=14)
                plt.ylim(0.5, 5.5)
                plt.yticks([1, 2, 3, 4, 5], ['1 - Low', '2', '3 - Medium', '4', '5 - High'])
                
                # Rotate date labels for better readability
                plt.xticks(rotation=45)
                plt.tight_layout()
            else:
                plt.text(0.5, 0.5, 'No mood data available yet', 
                        horizontalalignment='center', fontsize=14, color='#666')
                plt.axis('off')
            
            # Save plot to bytes buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Encode as base64 string
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
        except Exception as e:
            logger.error(f"Error generating mood chart: {str(e)}")
            return ""
    
    def generate_comparison_chart(self):
        """
        Generate performance comparison chart between Llama2 and Llama3
        
        Returns:
            str: Base64 encoded PNG image of the comparison chart
        """
        try:
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Metrics to compare
            metrics = [
                'Response Quality', 
                'Empathy', 
                'Personalization', 
                'Intent Accuracy'
            ]
            
            # Values for each model
            llama2_values = [
                self.comparison_metrics["llama2"]["response_quality"],
                self.comparison_metrics["llama2"]["empathy_score"],
                self.comparison_metrics["llama2"]["personalization"],
                self.comparison_metrics["llama2"]["intent_accuracy"]
            ]
            
            llama3_values = [
                self.comparison_metrics["llama3"]["response_quality"],
                self.comparison_metrics["llama3"]["empathy_score"],
                self.comparison_metrics["llama3"]["personalization"],
                self.comparison_metrics["llama3"]["intent_accuracy"]
            ]
            
            # Set width of bars
            barWidth = 0.35
            
            # Set positions of the bars on X axis
            r1 = np.arange(len(llama2_values))
            r2 = [x + barWidth for x in r1]
            
            # Create bars
            ax.bar(r1, llama2_values, width=barWidth, edgecolor='white', label='Llama2', color='#4285f4')
            ax.bar(r2, llama3_values, width=barWidth, edgecolor='white', label='Llama3', color='#34a853')
            
            # Add labels and custom grid
            ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
            ax.set_ylabel('Score (%)', fontweight='bold', fontsize=12)
            ax.set_title('Llama2 vs Llama3 Performance Comparison', fontsize=14)
            ax.set_xticks([r + barWidth/2 for r in range(len(llama2_values))])
            ax.set_xticklabels(metrics)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax.legend()
            
            # Save plot to bytes buffer
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Encode as base64 string
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
        except Exception as e:
            logger.error(f"Error generating comparison chart: {str(e)}")
            return ""
    
    def generate_response_time_comparison(self):
        """
        Generate response time comparison chart
        
        Returns:
            str: Base64 encoded PNG image
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Get average response times
            llama2_time = self.comparison_metrics["llama2"]["response_time"]
            llama3_time = self.comparison_metrics["llama3"]["response_time"]
            
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
            
            # Save plot to bytes buffer
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Encode as base64 string
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
        except Exception as e:
            logger.error(f"Error generating response time chart: {str(e)}")
            return ""

# Initialize global chatbot instance
chatbot = MentalHealthChatbot()

# Flask routes
@app.route('/')
def index():
    """Render the main chatbot interface"""
    # Initialize session data if needed
    if 'user_data' not in session:
        session['user_data'] = {
            'stress': 3,  # Default stress level
            'slot': None,  # No appointment by default
            'ai_enhanced': True,  # Use enhanced AI by default
            'active_model': 'llama3',  # Use Llama3 by default
            'last_login': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'mood_history': [(datetime.now().strftime("%Y-%m-%d"), 3)]
        }
    
    if 'conversation' not in session:
        # Initialize with welcome message
        session['conversation'] = [
            {
                'bot': "Hi! I'm your Mental Health Support Chatbot. How can I assist you today?"
            }
        ]
    
    user_data = session['user_data']
    
    # Determine stress class for UI
    stress_class = 'stress-medium'  # Default
    if 'stress' in user_data:
        if user_data['stress'] <= 2:
            stress_class = 'stress-low'
        elif user_data['stress'] <= 4:
            stress_class = 'stress-medium'
        else:
            stress_class = 'stress-high'
    
    # Prepare template variables
    return render_template(
        'index.html',
        stress_class=stress_class,
        stress_value=f"{user_data.get('stress', 3)}/5",
        appointment=user_data.get('slot', 'None'),
        ai_enhanced=user_data.get('ai_enhanced', True),
        active_model=user_data.get('active_model', 'llama3'),
        conversation=session['conversation']
    )

@app.route('/chat', methods=['POST'])
def chat():
    """Process chat messages"""
    user_input = request.form.get('user_input', '')
    
    if not user_input.strip():
        return jsonify({
            'bot_response': "I didn't receive a message. Could you please try again?",
            'stress': session['user_data'].get('stress', 3),
            'stress_class': 'stress-medium',
            'appointment': session['user_data'].get('slot', None),
            'ai_enhanced': session['user_data'].get('ai_enhanced', True)
        })
    
    # Add user message to conversation history
    if 'conversation' not in session:
        session['conversation'] = []
    
    session['conversation'].append({'user': user_input})
    session.modified = True
    
    # Process the input and generate response
    try:
        user_data = session['user_data']
        active_model = user_data.get('active_model', 'llama3')
        
        # Generate response
        response_data = chatbot.generate_response(
            user_input, 
            session['conversation'], 
            user_data,
            active_model
        )
        
        content = response_data['content']
        intent = response_data['intent']
        
        # Handle numeric-only responses (stress levels)
        if user_input.strip().isdigit() and 1 <= int(user_input) <= 5:
            stress_level = int(user_input)
            stress_class = chatbot.update_stress_level(stress_level, user_data)
            
            # Add acknowledgment to the response
            content = f"Thank you for sharing your stress level ({stress_level}/5). {content}"
        
        # Handle slot selection intent
        if intent == "slot_selection":
            slot = extract_time_slot(user_input, chatbot.available_slots)
            if slot:
                user_data["slot"] = slot.capitalize()
                # Ensure the response acknowledges the booking
                if "confirmed" not in content.lower() and "book" not in content.lower():
                    content = f"I've booked your appointment for {slot.capitalize()}. {content}"
        
        # Add the bot response to conversation history
        session['conversation'].append({'bot': content})
        session.modified = True
        
        # Update session data
        session['user_data'] = user_data
        
        # Get current stress class
        stress_class = 'stress-medium'  # Default
        if 'stress' in user_data:
            if user_data['stress'] <= 2:
                stress_class = 'stress-low'
            elif user_data['stress'] <= 4:
                stress_class = 'stress-medium'
            else:
                stress_class = 'stress-high'
        
        # Return the response data
        return jsonify({
            'bot_response': content,
            'stress': user_data.get('stress', 3),
            'stress_class': stress_class,
            'appointment': user_data.get('slot', None),
            'ai_enhanced': user_data.get('ai_enhanced', True),
            'active_model': user_data.get('active_model', 'llama3'),
            'response_time': f"{response_data['response_time']:.2f}s",
            'quality_score': f"{response_data['quality_score']*100:.1f}%",
            'template_used': response_data['template_used']
        })
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        error_message = "I'm having trouble processing your request. Please try again later."
        
        # Add error response to conversation
        session['conversation'].append({'bot': error_message})
        session.modified = True
        
        return jsonify({
            'bot_response': error_message,
            'stress': session['user_data'].get('stress', 3),
            'stress_class': 'stress-medium',
            'appointment': session['user_data'].get('slot', None),
            'ai_enhanced': session['user_data'].get('ai_enhanced', True)
        })

@app.route('/toggle-model', methods=['POST'])
def toggle_model():
    """Toggle between Llama2 and Llama3 models"""
    try:
        if 'user_data' not in session:
            session['user_data'] = {}
        
        # Get current model and toggle
        current_model = session['user_data'].get('active_model', 'llama3')
        new_model = 'llama2' if current_model == 'llama3' else 'llama3'
        
        # Update user data
        session['user_data']['active_model'] = new_model
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': f"Switched to {new_model.upper()} model",
            'active_model': new_model
        })
    except Exception as e:
        logger.error(f"Error toggling model: {str(e)}")
        return jsonify({
            'success': False,
            'message': "Error toggling model",
            'active_model': session['user_data'].get('active_model', 'llama3')
        })

@app.route('/toggle-ai', methods=['POST'])
def toggle_ai():
    """Toggle between basic and enhanced AI modes"""
    try:
        if 'user_data' not in session:
            session['user_data'] = {}
        
        # Toggle the AI mode
        current_mode = session['user_data'].get('ai_enhanced', True)
        session['user_data']['ai_enhanced'] = not current_mode
        session.modified = True
        
        new_mode = session['user_data']['ai_enhanced']
        message = "AI Enhanced Mode activated" if new_mode else "Basic Mode activated"
        
        return jsonify({
            'success': True,
            'message': message,
            'ai_enhanced': new_mode
        })
    except Exception as e:
        logger.error(f"Error toggling AI mode: {str(e)}")
        return jsonify({
            'success': False,
            'message': "Error toggling AI mode",
            'ai_enhanced': session['user_data'].get('ai_enhanced', True)
        })

@app.route('/mood-chart')
def mood_chart():
    """Generate and return mood chart"""
    if 'user_data' not in session or 'mood_history' not in session['user_data']:
        return jsonify({'image': ''})
    
    mood_history = session['user_data']['mood_history']
    image_data = chatbot.generate_mood_chart_image(mood_history)
    
    return jsonify({'image': image_data})

@app.route('/model-comparison')
def model_comparison():
    """Return model comparison data and charts"""
    # Generate comparison charts
    performance_chart = chatbot.generate_comparison_chart()
    response_time_chart = chatbot.generate_response_time_comparison()
    
    # Format metrics for display
    llama2_metrics = chatbot.comparison_metrics["llama2"]
    llama3_metrics = chatbot.comparison_metrics["llama3"]
    
    return jsonify({
        'performance_chart': performance_chart,
        'response_time_chart': response_time_chart,
        'llama2': {
            'response_quality': f"{llama2_metrics['response_quality']:.1f}%",
            'empathy_score': f"{llama2_metrics['empathy_score']:.1f}%",
            'personalization': f"{llama2_metrics['personalization']:.1f}%",
            'template_usage': f"{llama2_metrics['template_usage']:.1f}%",
            'response_time': f"{llama2_metrics['response_time']:.2f}s",
            'intent_accuracy': f"{llama2_metrics['intent_accuracy']:.1f}%"
        },
        'llama3': {
            'response_quality': f"{llama3_metrics['response_quality']:.1f}%",
            'empathy_score': f"{llama3_metrics['empathy_score']:.1f}%",
            'personalization': f"{llama3_metrics['personalization']:.1f}%",
            'template_usage': f"{llama3_metrics['template_usage']:.1f}%", 
            'response_time': f"{llama3_metrics['response_time']:.2f}s",
            'intent_accuracy': f"{llama3_metrics['intent_accuracy']:.1f}%"
        },
        'active_model': session['user_data'].get('active_model', 'llama3')
    })

@app.route('/clear-conversation', methods=['POST'])
def clear_conversation():
    """Clear the conversation history"""
    try:
        # Reset conversation but keep user data
        session['conversation'] = [
            {
                'bot': "Hi! I'm your Mental Health Support Chatbot. How can I assist you today?"
            }
        ]
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': "Conversation cleared"
        })
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        return jsonify({
            'success': False,
            'message': "Error clearing conversation"
        })


@app.route('/api/session-data')
def get_session_data():
    """API endpoint to return session data as JSON"""
    if 'user_data' not in session:
        session['user_data'] = {
            'stress': 3,
            'slot': None,
            'ai_enhanced': True,
            'active_model': 'llama3',
            'last_login': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'mood_history': [(datetime.now().strftime("%Y-%m-%d"), 3)]
        }
    
    # Debug log to see what's being sent
    logger.info(f"Sending session data: {session['user_data']}")
    
    # Return only the data needed by the frontend
    response_data = {
        'stress': session['user_data'].get('stress', 3),
        'appointment': session['user_data'].get('slot', 'None'),
        'aiEnhanced': bool(session['user_data'].get('ai_enhanced', True)),
        'activeModel': session['user_data'].get('active_model', 'llama3')
    }
    logger.info(f"Formatted response data: {response_data}")
    return jsonify(response_data)
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
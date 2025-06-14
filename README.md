# Mental Health Chatbot with LLM Integration

This project implements an advanced mental health chatbot using Flask, with support for different LLM models (Llama2 and Llama3) through Ollama integration. The system includes sophisticated intent recognition, personalized responses, advanced FAQ matching, and a built-in evaluation framework.

## Project Structure

- `app.py` - Main Flask web application
- `advanced_faq_matcher.py` - Enhanced FAQ matching with semantic similarity
- `enhanced_intent_recognition.py` - Intent recognition with context handling
- `enhanced_ollama_integration.py` - Integration with Llama2 and Llama3 models
- `enhanced_response_evaluator.py` - Response quality evaluation
- `model_specific_intent.py` - Model-specific intent recognition
- `run_chatbot_evaluation.py` - Evaluation script to compare model performance
- `simplified_model_tester.py` - Testing framework for model evaluation

## Prerequisites

This project requires Python 3.8 or higher. Before running the applications, please ensure your system has the following prerequisites:

1. **Python 3.8+** installed on your system
2. **pip** (Python package manager)
3. **Virtual environment** (recommended)
4. **Ollama** (optional, only if you want to use actual LLM models instead of simulation)

Installation
## Using Ollama for Actual LLM Integration
you need to install and configure Ollama:
1. Download and install Ollama from https://ollama.ai/
2. Pull the required models:
   ```bash
   ollama pull llama2
   ollama pull llama3
   ```
3. Run Ollama server:
   ```bash
   ollama serve
   ```
The application will automatically detect and use Ollama if it's running on the default port (11434).

### Step 1: Create and activate a virtual environment (recommended)
#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
### Step 2: Install the required packages
```bash
pip install -r requirements.txt
```

This will install all the necessary dependencies including:
- Flask (3.0.0+)
- pandas (2.1.0+)
- numpy (1.26.0+)
- scikit-learn (1.2.2)
- nltk (3.8.1)
- matplotlib (3.7.1)
- seaborn (0.12.2)
- requests (2.28.2)
- python-dotenv (1.0.0)

### Step 3: Download NLTK resources (if not downloaded automatically)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 4: Make sure the required directory structure exists
```bash
mkdir -p data results
```

## Running the Web Application
The main web application is implemented in `app.py`. To run it:

```bash
python app.py
```

This will start a Flask development server, typically on http://127.0.0.1:5000/. Open this URL in your web browser to interact with the chatbot.

### Notes about the web application:
- The application will use simulated LLM responses if Ollama is not available.
- You can toggle between Llama2 and Llama3 models in the interface.
- Performance metrics are displayed in the UI.
- The application provides stress tracking, appointment booking, and mental health resources.
## Running the Model Evaluation
To run the comprehensive evaluation comparing Llama2 and Llama3 models:

```bash
python run_chatbot_evaluation.py
```

This script will:
1. Run test cases against both models
2. Calculate detailed performance metrics
3. Generate visualizations comparing the models
4. Create an HTML report with the evaluation results
5. Generate a separate report for intent classification performance

### Notes about the evaluation:

- The evaluation will run in simulation mode if Ollama is not available
- Results and visualizations are saved in the `results` directory
- The main report is available at `results/llama_comparison_report.html`
- An intent classification report is saved at `results/intent_classification_report.md`

## Customizing the FAQ Database

The chatbot uses a CSV file for FAQ matching. You can customize this by creating a CSV file at `data/Mental_Health_FAQ.csv` with the following format:

```
Questions,Answers
"What is anxiety?","Anxiety is a normal emotion characterized by feelings of tension, worried thoughts, and physical changes..."
"How can I manage stress?","Several effective stress management techniques include deep breathing, meditation..."
```

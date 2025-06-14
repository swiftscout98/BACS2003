import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams

class AdvancedFAQMatcher:
    """Advanced FAQ matching with multiple strategies and semantic similarity"""
    
    def __init__(self, faq_data):
        """
        Initialize the FAQ matcher with FAQ data
        
        Args:
            faq_data (DataFrame): DataFrame containing FAQ data with 'Questions' and 'Answers' columns
        """
        # Download NLTK resources if needed
        self._ensure_nltk_resources()
        
        # Initialize text processing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Prepare FAQ data
        self.faq_df = faq_data
        # Convert Questions column to string before using str accessor
        self.questions = list(self.faq_df["Questions"].astype(str).str.lower())
        self.answers = list(self.faq_df["Answers"])
        
        # Create synonym mappings for mental health terms - moved up to fix initialization order
        self.synonyms = self._create_synonym_mapping()
        
        # Create expanded questions for better matching
        self.expanded_questions = self._expand_questions()
        
        # Create preprocessed versions
        self.preprocessed_questions = [self._preprocess_text(q) for q in self.questions]
        self.preprocessed_expanded = [self._preprocess_text(q) for q in self.expanded_questions]
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=1, 
            max_df=0.9,
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            sublinear_tf=True
        )
        
        # Create TF-IDF matrices
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_questions + self.preprocessed_expanded)
        
        # Create keyword index for faster lookup
        self.keyword_index = self._build_keyword_index()
    
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
                print(f"Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, quiet=True)
    
    def _expand_questions(self):
        """Expand questions with variations to improve matching"""
        expanded = []
        
        for question in self.questions:
            # Add the original question
            expanded.append(question)
            
            # Add variations
            # 1. Question with "how to" prefix if not already present
            if not question.startswith(("how", "what", "why", "when", "where", "who", "is", "are", "can")):
                expanded.append(f"how to {question}")
            
            # 2. Question with "what is" prefix for definitional questions
            if not question.startswith(("what is", "what are", "what's", "whats")):
                # Check if it might be a definitional question
                tokens = question.split()
                if len(tokens) <= 4:
                    expanded.append(f"what is {question}")
                    expanded.append(f"define {question}")
            
            # 3. Statement to question transformation
            if question.endswith("?"):
                # Already a question, make a statement
                statement = question[:-1].strip()
                expanded.append(statement)
            else:
                # Make it a question if it's not already
                expanded.append(f"{question}?")
        
        return expanded
    
    def _preprocess_text(self, text):
        """
        Preprocess text for better matching
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [t for t in tokens if t not in self.stop_words]
        
        # Apply synonym replacement for mental health terms
        synonym_tokens = self._apply_synonyms(filtered_tokens)
        
        # Lemmatize
        lemmatized_tokens = [self.lemmatizer.lemmatize(t) for t in synonym_tokens]
        
        return ' '.join(lemmatized_tokens)
    
    def _apply_synonyms(self, tokens):
        """Apply synonym replacement for mental health terms"""
        result = []
        for token in tokens:
            # Check if the token is in our synonym mapping
            if token in self.synonyms:
                # Use the canonical form
                result.append(self.synonyms[token])
            else:
                result.append(token)
        return result
    
    def _create_synonym_mapping(self):
        """Create mapping of synonyms to canonical forms for mental health terms"""
        return {
            # Anxiety terms
            "anxious": "anxiety",
            "nervous": "anxiety",
            "worried": "anxiety",
            "worrying": "anxiety",
            "panic": "anxiety",
            "fear": "anxiety",
            "phobia": "anxiety",
            
            # Depression terms
            "depressed": "depression",
            "sad": "depression",
            "unhappy": "depression",
            "melancholy": "depression",
            "despair": "depression",
            "hopeless": "depression",
            "hopelessness": "depression",
            
            # Stress terms
            "stressed": "stress",
            "stressful": "stress",
            "pressure": "stress",
            "overwhelmed": "stress",
            "burnout": "stress",
            "exhausted": "stress",
            "exhaustion": "stress",
            
            # Sleep terms
            "insomnia": "sleep",
            "sleepless": "sleep",
            "sleeplessness": "sleep",
            "tired": "sleep",
            "fatigue": "sleep",
            "rest": "sleep",
            "resting": "sleep",
            
            # Therapy terms
            "therapist": "therapy",
            "counselor": "therapy",
            "counseling": "therapy",
            "psychologist": "therapy",
            "psychiatrist": "therapy",
            "treatment": "therapy",
            
            # Medication terms
            "medicine": "medication",
            "drug": "medication",
            "prescription": "medication",
            "antidepressant": "medication",
            "pill": "medication",
            "meds": "medication"
        }
    
    def _build_keyword_index(self):
        """
        Build keyword index for faster lookups
        
        Returns:
            dict: Mapping from keywords to question indices
        """
        keyword_index = {}
        
        for idx, question in enumerate(self.questions + self.expanded_questions):
            # Extract content words (nouns, verbs, adjectives)
            tokens = word_tokenize(question.lower())
            
            # Filter out stopwords
            content_words = [w for w in tokens if w not in self.stop_words and len(w) > 2]
            
            # Create word stems for more robust matching
            stems = [self.stemmer.stem(w) for w in content_words]
            
            # Add unigrams to the index
            for word in content_words + stems:
                if word not in keyword_index:
                    keyword_index[word] = set()
                keyword_index[word].add(idx)
            
            # Add bigrams to the index for phrase matching
            if len(content_words) >= 2:
                bigrams_list = list(ngrams(content_words, 2))
                for bigram in bigrams_list:
                    bigram_key = f"{bigram[0]}_{bigram[1]}"
                    if bigram_key not in keyword_index:
                        keyword_index[bigram_key] = set()
                    keyword_index[bigram_key].add(idx)
        
        return keyword_index
    
    def find_match(self, user_input, threshold=0.6, top_n=3):
        """
        Find best matching questions and answers
        
        Args:
            user_input (str): User's question
            threshold (float): Similarity threshold (0-1)
            top_n (int): Number of top matches to consider
            
        Returns:
            tuple: (matched_question, answer, confidence_score) or (None, None, 0)
        """
        if not user_input or not user_input.strip():
            return None, None, 0
            
        # Preprocess the input
        preprocessed_input = self._preprocess_text(user_input)
        input_lower = user_input.lower().strip()
        
        # Strategy 1: Check for exact matches (case-insensitive)
        for i, question in enumerate(self.questions):
            similarity = self._calculate_string_similarity(input_lower, question.lower())
            if similarity > 0.9:  # Very high similarity
                return question, self.answers[i], 1.0
        
        # Strategy 2: Keyword-based filtering to reduce search space
        candidate_indices = self._get_candidate_indices(preprocessed_input, input_lower)
        
        if not candidate_indices:
            # If no candidates from keywords, use all questions
            candidate_indices = list(range(len(self.questions) + len(self.expanded_questions)))
        
        # Strategy 3: TF-IDF and cosine similarity for fuzzy matching
        user_vector = self.tfidf_vectorizer.transform([preprocessed_input])
        
        # Only calculate similarities with candidate questions
        candidate_matrix = self.tfidf_matrix[list(candidate_indices)]
        similarities = cosine_similarity(user_vector, candidate_matrix).flatten()
        
        # Find top N matches
        if len(similarities) > 0:
            # Sort indices by similarity score (descending)
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            # Filter by threshold
            filtered_indices = [idx for idx in top_indices if similarities[idx] >= threshold]
            
            if filtered_indices:
                best_idx = filtered_indices[0]
                best_score = similarities[best_idx]
                
                # Map back to original index
                original_idx = list(candidate_indices)[best_idx]
                
                # Determine if it's from original questions or expanded questions
                if original_idx < len(self.questions):
                    question_idx = original_idx
                else:
                    # Find the corresponding original question for expanded questions
                    expanded_idx = original_idx - len(self.questions)
                    question_idx = expanded_idx % len(self.questions)
                
                return self.questions[question_idx], self.answers[question_idx], best_score
        
        # No good match found
        return None, None, 0
    
    def _calculate_string_similarity(self, str1, str2):
        """Calculate string similarity using character-level matching"""
        # Simple character-level Jaccard similarity
        if not str1 or not str2:
            return 0
            
        set1 = set(str1)
        set2 = set(str2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0
    
    def _get_candidate_indices(self, preprocessed_input, original_input):
        """
        Get candidate question indices based on keyword matching
        
        Args:
            preprocessed_input (str): Preprocessed user input
            original_input (str): Original user input (lowercase)
            
        Returns:
            set: Set of candidate question indices
        """
        tokens = preprocessed_input.split()
        content_words = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        
        # Add stems for more robust matching
        stems = [self.stemmer.stem(w) for w in content_words]
        
        # Add mental health synonyms for better matching
        expanded_tokens = set(content_words + stems)
        for token in content_words:
            if token in self.synonyms:
                expanded_tokens.add(self.synonyms[token])
        
        # Collect all question indices that contain any of the input tokens
        candidate_indices = set()
        
        for token in expanded_tokens:
            if token in self.keyword_index:
                candidate_indices.update(self.keyword_index[token])
        
        # Add bigram matches for phrase matching
        if len(content_words) >= 2:
            bigrams_list = list(ngrams(content_words, 2))
            for bigram in bigrams_list:
                bigram_key = f"{bigram[0]}_{bigram[1]}"
                if bigram_key in self.keyword_index:
                    candidate_indices.update(self.keyword_index[bigram_key])
        
        return candidate_indices
    
    def get_top_matches(self, user_input, top_n=3):
        """
        Get top N matching questions and answers
        
        Args:
            user_input (str): User's question
            top_n (int): Number of top matches to return
            
        Returns:
            list: List of tuples (question, answer, score)
        """
        if not user_input or not user_input.strip():
            return []
            
        # Preprocess the input
        preprocessed_input = self._preprocess_text(user_input)
        
        # Get candidate indices
        candidate_indices = self._get_candidate_indices(
            preprocessed_input, user_input.lower().strip()
        )
        
        if not candidate_indices:
            # If no candidates from keywords, use all questions
            candidate_indices = list(range(len(self.questions) + len(self.expanded_questions)))
        
        # TF-IDF and cosine similarity
        user_vector = self.tfidf_vectorizer.transform([preprocessed_input])
        
        # Only calculate similarities with candidate questions
        candidate_matrix = self.tfidf_matrix[list(candidate_indices)]
        similarities = cosine_similarity(user_vector, candidate_matrix).flatten()
        
        # Find top N matches
        results = []
        if len(similarities) > 0:
            # Sort indices by similarity score (descending)
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            for idx in top_indices:
                score = similarities[idx]
                
                # Map back to original index
                original_idx = list(candidate_indices)[idx]
                
                # Determine if it's from original questions or expanded questions
                if original_idx < len(self.questions):
                    question_idx = original_idx
                else:
                    # Find the corresponding original question for expanded questions
                    expanded_idx = original_idx - len(self.questions)
                    question_idx = expanded_idx % len(self.questions)
                
                results.append((
                    self.questions[question_idx],
                    self.answers[question_idx],
                    float(score)
                ))
        
        return results
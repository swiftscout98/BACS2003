import os
import sys
import logging
import time
from datetime import datetime

# Make sure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced model tester
from simplified_model_tester import EnhancedModelTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chatbot_evaluator")

def main():
    """
    Run a comprehensive evaluation of the chatbot with both Llama2 and Llama3 models
    """
    print("=" * 80)
    print(f"CHATBOT EVALUATION: LLAMA2 VS LLAMA3 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    # Check if results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Initialize the tester with the advanced components
    print("\nInitializing evaluation framework...")
    faq_path = "data/Mental_Health_FAQ.csv"
    tester = EnhancedModelTester(faq_path)
    
    # Run the tests
    print("\nRunning model comparison tests...")
    start_time = time.time()
    llama2_results, llama3_results = tester.run_tests()
    elapsed_time = time.time() - start_time
    
    print(f"\nTests completed in {elapsed_time:.2f} seconds")
    print(f"Processed {len(llama2_results)} test cases for each model")
    
    # Calculate metrics
    print("\nCalculating detailed performance metrics...")
    llama2_metrics, llama3_metrics = tester.calculate_metrics(llama2_results, llama3_results)
    
    # Save results
    print("\nSaving detailed results...")
    metrics = tester.save_results(llama2_results, llama3_results, llama2_metrics, llama3_metrics)
    
    # Generate visualizations
    print("\nGenerating comparison visualizations...")
    tester.generate_visualizations(llama2_metrics, llama3_metrics)
    
    # Generate HTML report
    print("\nGenerating comprehensive evaluation report...")
    report_file = tester.generate_report(llama2_metrics, llama3_metrics)
    
    # Generate specialized intent classification report
    intent_report = "results/intent_classification_report.md"
    
    print("\n" + "=" * 40)
    print("Evaluation completed successfully!")
    print("=" * 40)
    
    print(f"\nMain report available at: {report_file}")
    print(f"Intent classification report available at: {intent_report}")
    print("Detailed results saved in results directory")
    
    # Print quick summary
    print("\nQUICK SUMMARY:")
    print(f"  Llama2 Overall Quality: {llama2_metrics.get('avg_quality_score', 0)*100:.1f}%")
    print(f"  Llama3 Overall Quality: {llama3_metrics.get('avg_quality_score', 0)*100:.1f}%")
    print(f"  Llama2 Intent Accuracy: {llama2_metrics.get('intent_accuracy', 0)*100:.1f}%")
    print(f"  Llama3 Intent Accuracy: {llama3_metrics.get('intent_accuracy', 0)*100:.1f}%")
    print(f"  Llama2 Empathy: {llama2_metrics.get('avg_empathy_score', 0)*100:.1f}%")
    print(f"  Llama3 Empathy: {llama3_metrics.get('avg_empathy_score', 0)*100:.1f}%")
    print(f"  Llama2 Response Time: {llama2_metrics.get('avg_response_time', 0):.2f}s")
    print(f"  Llama3 Response Time: {llama3_metrics.get('avg_response_time', 0):.2f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        print("Check chatbot_evaluation.log for details")
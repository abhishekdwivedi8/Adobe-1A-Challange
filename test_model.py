import os
import sys
import json
import shutil

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Modify paths for local testing
from app.main import process_all_pdfs
import app.main as main

def setup_test_environment():
    """Set up the test environment with local paths."""
    # Use local paths instead of Docker paths
    main.INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
    main.OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    main.MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'heading_model.joblib')
    
    # Create input and output directories if they don't exist
    os.makedirs(main.INPUT_DIR, exist_ok=True)
    os.makedirs(main.OUTPUT_DIR, exist_ok=True)
    
    # Copy sample PDFs to input directory for testing
    sample_pdf_dir = os.path.join(os.path.dirname(__file__), 'training', 'sample_pdfs')
    for pdf_file in os.listdir(sample_pdf_dir):
        if pdf_file.lower().endswith('.pdf'):
            src = os.path.join(sample_pdf_dir, pdf_file)
            dst = os.path.join(main.INPUT_DIR, pdf_file)
            shutil.copy2(src, dst)
    
    print(f"Copied {len(os.listdir(main.INPUT_DIR))} PDFs to input directory for testing.")

if __name__ == "__main__":
    print("Setting up test environment...")
    setup_test_environment()
    
    print("\nProcessing PDFs with trained model...")
    process_all_pdfs()
    
    # Check results
    output_files = [f for f in os.listdir(main.OUTPUT_DIR) if f.lower().endswith('.json')]
    print(f"\nGenerated {len(output_files)} JSON outlines:")
    
    for output_file in output_files:
        output_path = os.path.join(main.OUTPUT_DIR, output_file)
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n{output_file}:")
        print(f"  Title: {data['title']}")
        print(f"  Outline items: {len(data['outline'])}")
        
        # Print first few outline items
        for i, item in enumerate(data['outline'][:3]):
            print(f"    {item['level']} - {item['text'][:50]}...")
        
        if len(data['outline']) > 3:
            print(f"    ... and {len(data['outline']) - 3} more items")
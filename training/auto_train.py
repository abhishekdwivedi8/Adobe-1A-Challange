import os
import sys
import json
import random
import subprocess
from collections import defaultdict

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from app import features

def extract_sample_text_blocks(pdf_path, num_samples=5):
    """Extract sample text blocks from a PDF for labeling."""
    print(f"Extracting features from {os.path.basename(pdf_path)}...")
    blocks = features.extract_features_from_pdf(pdf_path)
    
    if not blocks:
        print(f"Warning: No text blocks found in {pdf_path}")
        return []
    
    # Filter out very short blocks and select a diverse set of samples
    filtered_blocks = [b for b in blocks if len(b['text'].strip()) > 10]
    
    # Sort blocks by font size z-score (descending) to prioritize potential headings
    filtered_blocks.sort(key=lambda x: x['features'][0], reverse=True)
    
    # Take top blocks as potential headings
    samples = []
    
    # Get potential title (largest font on first page)
    first_page_blocks = [b for b in filtered_blocks if b['page_num'] == 1]
    if first_page_blocks:
        samples.append({
            'file_name': os.path.basename(pdf_path),
            'page_number': 1,
            'text': first_page_blocks[0]['text'],
            'label': 'Title'
        })
    
    # Get potential H1 headings (large font, often centered)
    h1_candidates = [b for b in filtered_blocks 
                    if b['features'][0] > 1.0  # Large font
                    and b['features'][2] > 0.7  # Centered
                    and len(b['text'].split()) < 10]  # Not too long
    
    if h1_candidates:
        for i in range(min(2, len(h1_candidates))):
            samples.append({
                'file_name': os.path.basename(pdf_path),
                'page_number': h1_candidates[i]['page_num'],
                'text': h1_candidates[i]['text'],
                'label': 'H1'
            })
    
    # Get potential H2 headings (medium-large font)
    h2_candidates = [b for b in filtered_blocks 
                    if 0.5 < b['features'][0] < 1.0  # Medium-large font
                    and len(b['text'].split()) < 15  # Not too long
                    and b not in h1_candidates]
    
    if h2_candidates:
        for i in range(min(2, len(h2_candidates))):
            samples.append({
                'file_name': os.path.basename(pdf_path),
                'page_number': h2_candidates[i]['page_num'],
                'text': h2_candidates[i]['text'],
                'label': 'H2'
            })
    
    # Get potential H3 headings (slightly larger than body text, often with bullet points)
    h3_candidates = [b for b in filtered_blocks 
                    if 0.2 < b['features'][0] < 0.5  # Slightly larger than body
                    and b['features'][4] > 0  # Often starts with numbering
                    and len(b['text'].split()) < 20  # Not too long
                    and b not in h1_candidates and b not in h2_candidates]
    
    if h3_candidates:
        for i in range(min(2, len(h3_candidates))):
            samples.append({
                'file_name': os.path.basename(pdf_path),
                'page_number': h3_candidates[i]['page_num'],
                'text': h3_candidates[i]['text'],
                'label': 'H3'
            })
    
    # Get body text samples (normal font size, longer paragraphs)
    body_candidates = [b for b in filtered_blocks 
                      if abs(b['features'][0]) < 0.3  # Normal font size
                      and len(b['text'].split()) > 10  # Longer text
                      and b not in h1_candidates and b not in h2_candidates and b not in h3_candidates]
    
    if body_candidates:
        for i in range(min(3, len(body_candidates))):
            samples.append({
                'file_name': os.path.basename(pdf_path),
                'page_number': body_candidates[i]['page_num'],
                'text': body_candidates[i]['text'],
                'label': 'Body'
            })
    
    return samples

def main():
    """Main function to generate labeled data and train the model."""
    SAMPLE_PDF_DIR = 'training/sample_pdfs'
    LABELED_DATA_PATH = 'training/labeled_data.json'
    
    # Get all PDF files in the sample directory
    pdf_files = [f for f in os.listdir(SAMPLE_PDF_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the sample directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files for training.")
    
    # Extract sample text blocks from each PDF
    all_samples = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(SAMPLE_PDF_DIR, pdf_file)
        samples = extract_sample_text_blocks(pdf_path)
        all_samples.extend(samples)
    
    # Ensure we have a balanced dataset
    label_counts = defaultdict(int)
    for sample in all_samples:
        label_counts[sample['label']] += 1
    
    print("\nGenerated labeled data distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} samples")
    
    # Save the labeled data
    with open(LABELED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(all_samples)} labeled examples to {LABELED_DATA_PATH}")
    
    # Run the training script
    print("\nStarting model training...")
    subprocess.run(['python', 'training/train.py'], check=True)

if __name__ == "__main__":
    main()
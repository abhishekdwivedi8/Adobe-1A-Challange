import fitz  # PyMuPDF
import statistics
import re

def extract_features_from_pdf(pdf_path):
    """
    Main function to extract a list of features for each text block in a PDF.
    This version is more robust and uses the PyMuPDF library correctly.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return []

    # --- Step 1: Calculate the "Document DNA" ---
    all_font_sizes = []

    # Iterate through all pages and spans to gather font sizes accurately
    for page in doc:
        # The 'dict' output gives us the most detailed structure
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    all_font_sizes.append(round(span['size']))

    if not all_font_sizes:
        doc.close()
        return [] # Document has no text

    # Calculate statistics for the entire document
    mean_font_size = statistics.mean(all_font_sizes)
    std_dev_font_size = statistics.stdev(all_font_sizes) if len(all_font_sizes) > 1 else 0

    # --- Step 2: Extract features for each block using the Document DNA ---
    all_features = []
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_width = page.rect.width

        for block in page_dict.get("blocks", []):
            # Reconstruct the full text of the block
            block_text = ""
            # Collect properties for the block. We'll use the most common style.
            span_sizes = []
            span_flags = []

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span['text'] + " "
                    span_sizes.append(round(span['size']))
                    span_flags.append(span['flags'])

            block_text = block_text.strip()
            if not block_text:
                continue

            # Use the most common font size and style as representative for the block
            font_size = statistics.mode(span_sizes) if span_sizes else 0
            flags = statistics.mode(span_flags) if span_flags else 0
            is_bold = flags & 2**4  # The bold flag is the 4th bit

            # --- Calculate "Standout" Features ---

            # Feature 1: Relational Font Size (Z-score)
            font_size_zscore = (font_size - mean_font_size) / std_dev_font_size if std_dev_font_size > 0 else 0

            # Feature 2: Horizontal Centrality
            x0, _, x1, _ = block['bbox']
            block_width = x1 - x0
            block_center_x = x0 + (block_width / 2)
            page_center_x = page_width / 2
            horizontal_centrality = 1 - (abs(block_center_x - page_center_x) / page_center_x) if page_center_x > 0 else 0

            # Feature 3: Text-based features
            word_count = len(block_text.split())
            starts_with_numbering = 1 if re.match(r'^\s*(\d+[\.\d]|[A-Z][\.\)]|[ivxlc]+\.?|\•|\❖|\)\s+', block_text, re.IGNORECASE) else 0

            # Append all features and raw data for this block
            all_features.append({
                "text": block_text,
                "page_num": page_num + 1,
                # The actual features for the model
                "features": [
                    font_size_zscore,
                    float(is_bold),
                    horizontal_centrality,
                    word_count,
                    starts_with_numbering
                ]
            })

    doc.close()
    return all_features
import fitz  # PyMuPDF
import statistics
import re

def extract_features_from_pdf(pdf_path):
    """
    Main function to extract a list of features for each text block in a PDF.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return []

    all_font_sizes = []
    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    all_font_sizes.append(round(span['size']))

    if not all_font_sizes:
        doc.close()
        return []

    mean_font_size = statistics.mean(all_font_sizes)
    std_dev_font_size = statistics.stdev(all_font_sizes) if len(all_font_sizes) > 1 else 0

    all_features = []
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_width = page.rect.width

        for block in page_dict.get("blocks", []):
            block_text = ""
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
            
            font_size = statistics.mode(span_sizes) if span_sizes else 0
            flags = statistics.mode(span_flags) if span_flags else 0
            is_bold = flags & 2**4

            font_size_zscore = (font_size - mean_font_size) / std_dev_font_size if std_dev_font_size > 0 else 0
            
            x0, _, x1, _ = block['bbox']
            block_width = x1 - x0
            block_center_x = x0 + (block_width / 2)
            page_center_x = page_width / 2
            horizontal_centrality = 1 - (abs(block_center_x - page_center_x) / page_center_x) if page_center_x > 0 else 0
            
            word_count = len(block_text.split())
            starts_with_numbering = 1 if re.match(r'^\s*(\d+[\.\d]|[A-Z][\.\)]|[ivxlc]+\.?|\•|\❖|\)\s+)', block_text, re.IGNORECASE) else 0

            all_features.append({
                "text": block_text,
                "page_num": page_num + 1,
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
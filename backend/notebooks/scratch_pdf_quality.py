"""
Sample PDF text extraction quality check.
Uses pdfminer.six (lightweight, no ML).
Samples 2 PDFs from 3 different categories.
"""
import sys
import subprocess

# Install pdfminer.six into venv if not present
try:
    import pdfminer
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pdfminer.six', '-q'])

from pathlib import Path
import random
random.seed(42)

try:
    from pdfminer.high_level import extract_text
    HAVE_PDFMINER = True
except ImportError:
    HAVE_PDFMINER = False
    print('pdfminer not available')

RAW_DATA_DIR = Path('backend/data/raw/data')

# Sample 2 PDFs from each of 5 diverse categories
sample_cats = ['INFORMATION-TECHNOLOGY', 'HEALTHCARE', 'FINANCE', 'CHEF', 'ENGINEERING']
sample_files = []
for cat in sample_cats:
    cat_dir = RAW_DATA_DIR / cat
    if cat_dir.exists():
        pdfs = list(cat_dir.glob('*.pdf'))
        chosen = random.sample(pdfs, min(2, len(pdfs)))
        sample_files.extend([(cat, p) for p in chosen])

print(f'Sampling {len(sample_files)} PDFs from {len(sample_cats)} categories\n')

for cat, pdf_path in sample_files:
    print(f'=== {cat} | {pdf_path.name} ===')
    if not HAVE_PDFMINER:
        print('  [pdfminer not available — skipping extraction]')
        continue
    try:
        text = extract_text(str(pdf_path))
        text_clean = text.strip()
        n_chars = len(text_clean)
        n_words = len(text_clean.split())
        # Check for joined words (simple heuristic: very long tokens)
        words = text_clean.split()
        long_tokens = [w for w in words if len(w) > 25][:5]
        preview = text_clean[:400].replace('\n', ' ')
        print(f'  chars={n_chars}  words={n_words}')
        print(f'  Long tokens (>25c): {long_tokens}')
        print(f'  Preview: {preview}')
    except Exception as e:
        print(f'  ERROR: {e}')
    print()

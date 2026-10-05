"""
Overlap investigation: compare CSV IDs vs PDF filenames in data/ folders.
Also inspect a small sample of PDFs for text extraction quality.
"""
import os
import re
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path('backend/data/raw/data')
CSV_PATH     = Path('backend/data/raw/Resume.csv')

# --- Load CSV IDs ---
df = pd.read_csv(CSV_PATH, usecols=['ID', 'Category', 'Resume_str'],
                 encoding='utf-8', encoding_errors='replace')
csv_ids = set(df['ID'].astype(str).tolist())
csv_cats = set(df['Category'].tolist())
print(f'CSV rows: {len(df)}  | unique IDs: {len(csv_ids)}')
print(f'CSV categories: {sorted(csv_cats)}')
print()

# --- Collect all PDF filenames ---
all_files = []
for cat_dir in sorted(RAW_DATA_DIR.iterdir()):
    if cat_dir.is_dir():
        for f in cat_dir.iterdir():
            if f.is_file():
                all_files.append({'category': cat_dir.name, 'filename': f.name, 'stem': f.stem, 'ext': f.suffix.lower()})

files_df = pd.DataFrame(all_files)
print(f'Total PDF files: {len(files_df)}')
print()

# --- Check if stems are numeric (potential IDs) ---
files_df['stem_is_numeric'] = files_df['stem'].str.isnumeric()
n_numeric = files_df['stem_is_numeric'].sum()
print(f'Filenames with numeric stems: {n_numeric} / {len(files_df)}')
print(f'Sample filenames: {files_df["filename"].head(10).tolist()}')
print()

# --- If numeric stems, compare with CSV IDs ---
if n_numeric > 0:
    numeric_stems = set(files_df[files_df['stem_is_numeric']]['stem'].tolist())
    overlap_ids = csv_ids.intersection(numeric_stems)
    print(f'Numeric stems in folders:  {len(numeric_stems)}')
    print(f'CSV IDs:                   {len(csv_ids)}')
    print(f'EXACT ID matches:          {len(overlap_ids)}')
    if len(overlap_ids) > 0:
        print(f'Sample matching IDs: {list(overlap_ids)[:10]}')
    else:
        print('No exact ID matches found.')
else:
    print('Filenames are NOT numeric — cannot directly match to CSV IDs by filename.')
    print('Sample stems:', files_df['stem'].head(10).tolist())

print()
# --- Category name comparison ---
folder_cats = set(files_df['category'].tolist())
print('=== CATEGORY COMPARISON ===')
print(f'CSV categories  ({len(csv_cats)}): {sorted(csv_cats)}')
print(f'Folder categories ({len(folder_cats)}): {sorted(folder_cats)}')
exact_match = csv_cats.intersection(folder_cats)
only_csv    = csv_cats - folder_cats
only_folder = folder_cats - csv_cats
print(f'Exact matches:           {sorted(exact_match)}')
print(f'Only in CSV:             {sorted(only_csv)}')
print(f'Only in folders:         {sorted(only_folder)}')

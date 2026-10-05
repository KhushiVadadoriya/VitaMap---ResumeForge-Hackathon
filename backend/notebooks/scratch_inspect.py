import pandas as pd
import numpy as np

print('Loading full CSV...')
df = pd.read_csv('backend/data/raw/Resume.csv', encoding='utf-8', encoding_errors='replace')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print()

print('=== DTYPES ===')
print(df.dtypes)
print()

print('=== MISSING VALUES ===')
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(3)
for c in df.columns:
    print(f'  {c}: {miss[c]} nulls  ({miss_pct[c]}%)')
print()

n_dup_rows = df.duplicated().sum()
print(f'Duplicate complete rows: {n_dup_rows}')

id_unique = df['ID'].nunique()
print(f'ID unique: {id_unique} (total rows={len(df)})')
print(f'ID min={df["ID"].min()}  max={df["ID"].max()}')
print()

cat_vc = df['Category'].value_counts(dropna=False)
n_unique_cat = df['Category'].nunique()
print(f'=== CATEGORY unique={n_unique_cat} ===')
print(cat_vc.to_string())
print()

text = df['Resume_str'].astype(str)
n_null = df['Resume_str'].isnull().sum()
n_empty = (text == '').sum()
n_ws = (text.str.strip() == '').sum()
VERY_SHORT = 50
n_short = (text.str.strip().str.len() < VERY_SHORT).sum()
n_dup_text = df['Resume_str'].duplicated().sum()
tlen = text.str.len()

print('=== Resume_str QUALITY ===')
print(f'  Null:          {n_null}')
print(f'  Empty:         {n_empty}')
print(f'  Whitespace:    {n_ws}')
print(f'  Very short (<{VERY_SHORT}c): {n_short}')
print(f'  Dup text:      {n_dup_text}')
print()

print('=== TEXT LENGTH STATS (chars) ===')
print(f'  count  : {len(tlen)}')
print(f'  mean   : {tlen.mean():.1f}')
print(f'  median : {tlen.median():.1f}')
print(f'  std    : {tlen.std():.1f}')
print(f'  min    : {int(tlen.min())}')
print(f'  p5     : {tlen.quantile(0.05):.1f}')
print(f'  p25    : {tlen.quantile(0.25):.1f}')
print(f'  p75    : {tlen.quantile(0.75):.1f}')
print(f'  p95    : {tlen.quantile(0.95):.1f}')
print(f'  max    : {int(tlen.max())}')
print()

pd.set_option('display.max_colwidth', 120)
print('=== FIRST 3 ROWS (Resume_str[:200]) ===')
for i in range(3):
    row = df.iloc[i]
    print(f'  Row {i}: ID={row["ID"]}  Category={row["Category"]}')
    print(f'    Resume_str[:200]: {str(row["Resume_str"])[:200]}')
    print()

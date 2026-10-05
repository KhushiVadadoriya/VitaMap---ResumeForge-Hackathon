import pandas as pd
from collections import Counter
import re

df = pd.read_csv('backend/data/raw/Resume.csv', encoding='utf-8', encoding_errors='replace')

def simple_tokenize(text):
    return re.findall(r'[\w][\w.+#]*', str(text).lower())

# Top 20 tokens
all_tokens = []
for t in df['Resume_str'].astype(str):
    all_tokens.extend(simple_tokenize(t))
freq = Counter(all_tokens)
print('TOP 20 TOKENS:')
for w, c in freq.most_common(20):
    print(f'  {w:20s} {c:8d}')
print()

# Top 10 bigrams
all_bigrams = []
for t in df['Resume_str'].astype(str):
    toks = simple_tokenize(t)
    all_bigrams.extend(zip(toks[:-1], toks[1:]))
bfreq = Counter(all_bigrams)
print('TOP 10 BIGRAMS:')
for (w1,w2), c in bfreq.most_common(10):
    print(f'  {w1} {w2:30s} {c:8d}')
print()

# Technical term presence
technical = ['python','java','sql','aws','c++','.net','tensorflow',
             'nlp','machine','learning','deep','hadoop','spark','excel',
             'javascript','html','css','linux','docker','git']
print('TECHNICAL TERM FREQUENCIES:')
for t in technical:
    print(f'  {t:20s} {freq.get(t,0):8d}')
print()

# Categories with very short text
df['_tlen'] = df['Resume_str'].astype(str).str.len()
print('CATEGORIES — mean text length:')
print(df.groupby('Category')['_tlen'].agg(['mean','median','min','max']).sort_values('mean').to_string())

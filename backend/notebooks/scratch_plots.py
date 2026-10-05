"""
Phase 1 EDA plots — all plots for the structured dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
from pathlib import Path

RAW_CSV = Path('backend/data/raw/Resume.csv')
EDA_DIR = Path('backend/data/processed/eda')
EDA_DIR.mkdir(parents=True, exist_ok=True)

print('Loading CSV...')
df = pd.read_csv(RAW_CSV, encoding='utf-8', encoding_errors='replace')
print(f'Shape: {df.shape}')

TEXT_COL   = 'Resume_str'
TARGET_COL = 'Category'

# ── 1. CLASS DISTRIBUTION ────────────────────────────────────────────────────
cat_vc  = df[TARGET_COL].value_counts()
cat_pct = (cat_vc / len(df) * 100).round(2)

fig, ax = plt.subplots(figsize=(14, 8))
colors = sns.color_palette('tab20', n_colors=len(cat_vc))
bars = ax.barh(cat_vc.index[::-1], cat_vc.values[::-1], color=colors[::-1], edgecolor='white')
for bar, val in zip(bars, cat_vc.values[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(val), va='center', fontsize=9)
ax.set_xlabel('Count', fontsize=12)
ax.set_title(f'Class Distribution — {len(cat_vc)} Categories (n={len(df)})',
             fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
out = EDA_DIR / 'class_distribution.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── 2. TEXT LENGTH DISTRIBUTION ──────────────────────────────────────────────
df['_text_len'] = df[TEXT_COL].astype(str).str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['_text_len'], bins=60, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(df['_text_len'].median(), color='red', linestyle='--',
                label=f'Median={df["_text_len"].median():.0f}')
axes[0].axvline(df['_text_len'].mean(),   color='orange', linestyle='--',
                label=f'Mean={df["_text_len"].mean():.0f}')
axes[0].set_xlabel('Text Length (chars)')
axes[0].set_ylabel('Count')
axes[0].set_title('Resume Text Length — Histogram')
axes[0].legend()
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

df['_text_len'].plot.kde(ax=axes[1], color='steelblue')
axes[1].set_xlabel('Text Length (chars)')
axes[1].set_title('Resume Text Length — KDE')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('Resume Text Length Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
out = EDA_DIR / 'text_length_distribution.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── 3. TEXT LENGTH BY CATEGORY ───────────────────────────────────────────────
order = df.groupby(TARGET_COL)['_text_len'].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(12, 9))
sns.boxplot(data=df, y=TARGET_COL, x='_text_len', order=order,
            palette='tab20', orient='h', ax=ax, width=0.6,
            flierprops=dict(marker='o', markersize=2, alpha=0.4))
ax.set_xlabel('Text Length (chars)')
ax.set_ylabel('')
ax.set_title('Resume Text Length by Category', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
out = EDA_DIR / 'text_length_by_category.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── 4. TOKEN FREQUENCIES ─────────────────────────────────────────────────────
def simple_tokenize(text):
    return re.findall(r'[\w][\w.+#]*', str(text).lower())

print('Tokenizing...')
all_tokens = []
for t in df[TEXT_COL].astype(str):
    all_tokens.extend(simple_tokenize(t))

token_freq = Counter(all_tokens)
print(f'Total tokens: {len(all_tokens):,}  Unique: {len(token_freq):,}')

top30 = token_freq.most_common(30)
words30, counts30 = zip(*top30)

fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette('Blues_d', n_colors=30)
ax.barh(list(words30)[::-1], list(counts30)[::-1], color=colors)
ax.set_xlabel('Frequency')
ax.set_title('Top 30 Tokens — All Resumes\n(no stopword removal — EDA only)',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
out = EDA_DIR / 'top_words.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── 5. TOP BIGRAMS ───────────────────────────────────────────────────────────
print('Computing bigrams...')
all_bigrams = []
for t in df[TEXT_COL].astype(str):
    toks = simple_tokenize(t)
    all_bigrams.extend(zip(toks[:-1], toks[1:]))

bigram_freq = Counter(all_bigrams)
top20_bi = bigram_freq.most_common(20)
bi_labels = [f'{w1} {w2}' for (w1, w2), _ in top20_bi]
bi_counts = [c for _, c in top20_bi]

fig, ax = plt.subplots(figsize=(12, 7))
colors = sns.color_palette('Greens_d', n_colors=20)
ax.barh(bi_labels[::-1], bi_counts[::-1], color=colors)
ax.set_xlabel('Frequency')
ax.set_title('Top 20 Bigrams — All Resumes\n(no stopword removal — EDA only)',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
out = EDA_DIR / 'top_bigrams.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── 6. WORDCLOUD ─────────────────────────────────────────────────────────────
print('Generating wordcloud...')
full_text = ' '.join(df[TEXT_COL].astype(str).tolist())
wc = WordCloud(width=1200, height=600, background_color='white',
               colormap='viridis', max_words=200,
               collocations=False, random_state=42).generate(full_text)

fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('WordCloud — All Resume Texts (EDA only)', fontsize=14, fontweight='bold')
plt.tight_layout()
out = EDA_DIR / 'wordcloud.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── 7. FOLDER / CATEGORY FILE COUNTS ────────────────────────────────────────
from pathlib import Path as P

RAW_DATA_DIR = P('backend/data/raw/data')
cat_counts = {}
for cat_dir in sorted(RAW_DATA_DIR.iterdir()):
    if cat_dir.is_dir():
        cat_counts[cat_dir.name] = len(list(cat_dir.glob('*.pdf')))

cat_s = pd.Series(cat_counts).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette('tab20', n_colors=len(cat_s))
ax.barh(cat_s.index[::-1], cat_s.values[::-1], color=colors[::-1], edgecolor='white')
for i, (cat, val) in enumerate(zip(cat_s.index[::-1], cat_s.values[::-1])):
    ax.text(val + 0.3, i, str(val), va='center', fontsize=9)
ax.set_xlabel('PDF File Count')
ax.set_title('PDF Files per Category Folder', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
out = EDA_DIR / 'folder_category_counts.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

print('\nAll plots saved.')
